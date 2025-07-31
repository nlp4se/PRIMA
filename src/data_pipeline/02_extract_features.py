import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.config import *

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, input_dir: Path = FILTERED_RELEASES_DIR,
                 output_dir: Path = FEATURES_DIR):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_stats = {
            'apps_processed': 0,
            'releases_processed': 0,
            'features_extracted': {},
            'missing_data_counts': defaultdict(int),
            'extraction_errors': []
        }

    def extract_general_features(self, release: Dict) -> Dict[str, Any]:
        raw_data = release.get('raw_data') or {}
        release_data = raw_data.get('release_data') or {}

        features = {
            'release_name_length': len(release.get('name') or ''),
            'has_release_name': bool((release.get('name') or '').strip()),
            'tag_name_length': len(release_data.get('tag_name') or ''),
            'has_tag_name': bool((release_data.get('tag_name') or '').strip()),
            'body_length': len(release_data.get('body') or ''),
            'has_body': bool((release_data.get('body') or '').strip()),
            'body_line_count': len((release_data.get('body') or '').split('\n')),
            'num_assets': len(release_data.get('assets') or []),
            'has_assets': len(release_data.get('assets') or []) > 0,
            'is_draft': release_data.get('draft', False),
            'is_prerelease': release_data.get('prerelease', False),
            'sequence_number': release.get('sequence_number', 0),
            'is_first_release': release.get('is_first_release', False),
            'is_last_release': release.get('is_last_release', False)
        }

        return features

    def extract_apk_metrics(self, release: Dict) -> Dict[str, Any]:
        raw_data = release.get('raw_data') or {}
        apk_data = raw_data.get('apk_metrics') or {}
        android_manifest = apk_data.get('androidManifest') or {}

        features = {
            'apk_file_size': self._safe_int(apk_data.get('fileSize')),
            'apk_dex_size': self._safe_int(apk_data.get('dexSize')),
            'apk_arsc_size': self._safe_int(apk_data.get('arscSize')),
            'num_activities': self._safe_int(android_manifest.get('numberOfActivities')),
            'num_services': self._safe_int(android_manifest.get('numberOfServices')),
            'num_content_providers': self._safe_int(android_manifest.get('numberOfContentProviders')),
            'num_broadcast_receivers': self._safe_int(android_manifest.get('numberOfBroadcastReceivers')),
            'num_permissions': len(android_manifest.get('usesPermissions') or []),
            'num_libraries': len(android_manifest.get('usesLibrary') or []),
            'total_components': (
                    self._safe_int(android_manifest.get('numberOfActivities')) +
                    self._safe_int(android_manifest.get('numberOfServices')) +
                    self._safe_int(android_manifest.get('numberOfContentProviders')) +
                    self._safe_int(android_manifest.get('numberOfBroadcastReceivers'))
            ),
            'has_apk_data': bool(apk_data),
            'min_sdk_version': self._safe_int(android_manifest.get('usesMinSdkVersion')),
            'target_sdk_version': self._safe_int(android_manifest.get('usesTargetSdkVersion')),
        }

        return features

    def extract_issue_features(self, release: Dict) -> Dict[str, Any]:
        raw_data = release.get('raw_data') or {}
        created_issues = raw_data.get('created_issues') or []
        closed_issues = raw_data.get('closed_issues') or []

        features = {
            'issue_created_count': len(created_issues),
            'issue_closed_count': len(closed_issues),
            'issue_net_count': len(created_issues) - len(closed_issues),
            'issue_has_activity': bool(created_issues or closed_issues),
            'issue_resolution_rate': len(closed_issues) / max(1, len(created_issues)) if created_issues else 0,
            'issue_avg_comments_created': 0.0,
            'issue_avg_comments_closed': 0.0
        }

        if created_issues:
            features['issue_avg_comments_created'] = np.mean([
                self._safe_int(issue.get('comments')) for issue in created_issues
            ])

        if closed_issues:
            features['issue_avg_comments_closed'] = np.mean([
                self._safe_int(issue.get('comments')) for issue in closed_issues
            ])

        return features

    def extract_pr_features(self, release: Dict) -> Dict[str, Any]:
        raw_data = release.get('raw_data') or {}
        pull_requests = raw_data.get('pull_requests') or []

        if not pull_requests:
            return {
                'pr_count': 0,
                'pr_merged_count': 0,
                'pr_closed_count': 0,
                'pr_merge_rate': 0.0,
                'pr_has_activity': False,
                'pr_avg_reviews': 0.0
            }

        merged_count = 0
        closed_count = 0
        review_counts = []

        for pr in pull_requests:
            if pr.get('merged_at'):
                merged_count += 1
            if pr.get('closed_at'):
                closed_count += 1

            reviews = pr.get('reviews') or []
            review_counts.append(len(reviews) if isinstance(reviews, list) else 0)

        return {
            'pr_count': len(pull_requests),
            'pr_merged_count': merged_count,
            'pr_closed_count': closed_count,
            'pr_merge_rate': merged_count / max(1, len(pull_requests)),
            'pr_has_activity': True,
            'pr_avg_reviews': np.mean(review_counts) if review_counts else 0.0
        }

    def extract_contributor_features(self, release: Dict) -> Dict[str, Any]:
        raw_data = release.get('raw_data') or {}
        contributors = raw_data.get('contributors') or []

        return {
            'contributor_count': len(contributors),
            'has_contributors': len(contributors) > 0
        }

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def extract_temporal_features(self, release: Dict, previous_release: Optional[Dict] = None) -> Dict[str, Any]:
        current_date = release.get("date")
        prev_date = previous_release.get("date") if previous_release else None

        # Ensure both are datetime objects
        if isinstance(current_date, str):
            try:
                current_date = pd.to_datetime(current_date)
            except Exception:
                current_date = None

        if isinstance(prev_date, str):
            try:
                prev_date = pd.to_datetime(prev_date)
            except Exception:
                prev_date = None

        days_since_last = (current_date - prev_date).days if current_date and prev_date else None

        return {
            'days_since_last_release': days_since_last
        }

    def extract_linguistic_features(self, release: Dict) -> Dict[str, Any]:
        release_data = release.get('raw_data', {}).get('release_data') or {}
        body = release_data.get('body') or ""
        words = body.split()
        sentences = body.split('.') if body else []

        features = {
            'body_char_count': len(body),
            'body_word_count': len(words),
            'body_line_count': len(body.splitlines()),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0.0,
            'sentence_count': len([s for s in sentences if s.strip()]),
            'has_numbers': int(any(char.isdigit() for char in body))
        }
        return features

    def extract_release_features(self, release: Dict, previous_release: Optional[Dict] = None) -> Dict[str, Any]:
        features = {}

        try:
            features.update(self.extract_general_features(release))
            features.update(self.extract_apk_metrics(release))
            features.update(self.extract_issue_features(release))
            features.update(self.extract_pr_features(release))
            features.update(self.extract_contributor_features(release))
            features.update(self.extract_linguistic_features(release))
            features.update(self.extract_temporal_features(release, previous_release))

            features['app_id'] = release['app_id']
            features['release_id'] = release['release_id']
            features['date'] = release['date']
            features['review_count'] = release['review_count']
            features['average_rating'] = release.get('average_rating', np.nan)

        except Exception as e:
            logger.error(f"Error extracting features for {release['app_id']}/{release['release_id']}: {e}")
            self.extraction_stats['extraction_errors'].append({
                'app_id': release['app_id'],
                'release_id': release['release_id'],
                'error': str(e)
            })

            features = {
                'app_id': release['app_id'],
                'release_id': release['release_id'],
                'date': release['date'],
                'review_count': release['review_count'],
                'average_rating': release.get('average_rating', np.nan)
            }

        return features

    def process_app(self, app_file: Path) -> List[Dict[str, Any]]:
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                app_data = json.load(f)

            releases = sorted(app_data['releases'], key=lambda r: r['date'])  # Ensure order
            app_features = []

            for i, release in enumerate(releases):
                prev_release = releases[i - 1] if i > 0 else None
                features = self.extract_release_features(release, prev_release)
                app_features.append(features)
                self.extraction_stats['releases_processed'] += 1

            self.extraction_stats['apps_processed'] += 1
            logger.info(f"Processed {len(releases)} releases for {app_data['app_id']}")

            return app_features

        except Exception as e:
            logger.error(f"Error processing app file {app_file}: {e}")
            return []

    def run_feature_extraction(self) -> pd.DataFrame:
        logger.info("Starting feature extraction pipeline")

        app_files_dir = self.input_dir / "individual_apps"
        app_files = list(app_files_dir.glob("*.json"))

        logger.info(f"Found {len(app_files)} app files to process")

        all_features = []
        processed_count = 0

        for i, app_file in enumerate(app_files):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(app_files)} apps processed")

            app_features = self.process_app(app_file)
            all_features.extend(app_features)

            if app_features:
                processed_count += 1

        logger.info(f"Feature extraction complete:")
        logger.info(f"  Apps processed: {processed_count}")
        logger.info(f"  Releases processed: {len(all_features)}")
        logger.info(f"  Extraction errors: {len(self.extraction_stats['extraction_errors'])}")

        features_df = pd.DataFrame(all_features)

        features_csv = self.output_dir / "extracted_features.csv"
        features_df.to_csv(features_csv, index=False)

        feature_metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_releases': len(features_df),
            'total_apps': len(features_df['app_id'].unique()),
            'feature_count': len(features_df.columns) - 4,
            'feature_names': list(features_df.columns),
            'extraction_stats': dict(self.extraction_stats)
        }

        metadata_file = self.output_dir / "feature_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(feature_metadata, f, indent=2, default=str)

        logger.info(f"Feature extraction completed!")
        logger.info(f"   Features extracted: {len(features_df)} releases")
        logger.info(f"   Feature dimensions: {len(features_df.columns)} columns")
        logger.info(f"   Saved to: {features_csv}")

        return features_df


def main():

    extractor = FeatureExtractor()
    features_df = extractor.run_feature_extraction()

    print(f"\nFeature extraction complete!")
    print(f"Extracted {len(features_df)} release feature vectors")
    print(f"Results saved in: {FEATURES_DIR}")

    return features_df


if __name__ == "__main__":
    main()