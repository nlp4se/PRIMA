import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pandas as pd

from src.utils.utils import get_app_id_from_filename
from src.utils.config import *

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ReleaseFilter:
    def __init__(self, input_dir: Path = RELEASE_JSON_DIR,
                 output_dir: Path = FILTERED_RELEASES_DIR):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'total_files': 0,
            'total_releases': 0,
            'releases_with_reviews': 0,
            'apps_processed': 0,
            'apps_kept': 0,
            'final_releases': 0,
            'apps_by_release_count': Counter(),
            'releases_by_review_count': Counter()
        }

    def load_release_data(self, file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                if len(data) == 0:
                    logger.debug(f"Empty release list in {file_path}")
                    return None
                return {'releases': data}
            elif isinstance(data, dict) and 'releases' in data:
                return data
            else:
                logger.warning(f"Invalid format in {file_path}: not a list or missing 'releases'")
                return None

        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def parse_release_date(self, release: Dict) -> Optional[datetime]:
        release_data = release.get('release_data', {})
        date_fields = ['created_at', 'published_at', 'date']

        for field in date_fields:
            if field in release_data and release_data[field]:
                try:
                    date_str = release_data[field]
                    if isinstance(date_str, str):
                        if 'T' in date_str:
                            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y'):
                                try:
                                    return datetime.strptime(date_str, fmt)
                                except ValueError:
                                    continue
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse date {date_str}: {e}")
                    continue

        for field in date_fields:
            if field in release and release[field]:
                try:
                    date_str = release[field]
                    if isinstance(date_str, str):
                        if 'T' in date_str:
                            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y'):
                                try:
                                    return datetime.strptime(date_str, fmt)
                                except ValueError:
                                    continue
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse date {date_str}: {e}")
                    continue

        logger.warning("No valid date found in release")
        return None

    def count_reviews(self, release: Dict) -> int:
        if 'google_play_reviews' in release:
            reviews = release['google_play_reviews']
            if isinstance(reviews, list):
                return len(reviews)
            elif isinstance(reviews, int):
                return reviews

        if 'reviews' in release:
            reviews = release['reviews']
            if isinstance(reviews, list):
                return len(reviews)
            elif isinstance(reviews, dict):
                return len(reviews)
            elif isinstance(reviews, int):
                return reviews

        return 0

    def process_app_releases(self, app_data: Dict, app_id: str) -> Tuple[List[Dict], Dict]:
        releases = app_data.get('releases', [])
        processed_releases = []
        app_stats = {
            'total_releases': len(releases),
            'releases_with_reviews': 0,
            'total_reviews': 0,
            'date_range': None
        }

        valid_releases = []

        for release in releases:
            release_date = self.parse_release_date(release)
            if not release_date:
                logger.debug(f"Skipping release without valid date in {app_id}")
                continue

            review_count = self.count_reviews(release)

            release_data = release.get('release_data', release)

            processed_release = {
                'app_id': app_id,
                'release_id': release_data.get('id', release_data.get('tag_name', 'unknown')),
                'name': release_data.get('name', release_data.get('tag_name', '')),
                'date': release_date,
                'review_count': review_count,
                'raw_data': release
            }

            valid_releases.append(processed_release)

            if review_count > 0:
                app_stats['releases_with_reviews'] += 1
                app_stats['total_reviews'] += review_count

        valid_releases.sort(key=lambda x: x['date'])

        for i, release in enumerate(valid_releases):
            release['sequence_number'] = i
            release['is_first_release'] = (i == 0)
            release['is_last_release'] = (i == len(valid_releases) - 1)

        if valid_releases:
            app_stats['date_range'] = (
                valid_releases[0]['date'],
                valid_releases[-1]['date']
            )

        return valid_releases, app_stats

    def filter_apps_by_criteria(self, all_apps_data: Dict) -> Dict:
        filtered_apps = {}

        for app_id, (releases, stats) in all_apps_data.items():
            if len(releases) < MIN_RELEASES_PER_APP:
                logger.debug(f"App {app_id}: insufficient releases ({len(releases)} < {MIN_RELEASES_PER_APP})")
                continue

            releases_with_reviews = sum(1 for r in releases if r['review_count'] > 0)
            if releases_with_reviews < MIN_RELEASES_WITH_REVIEWS:
                logger.debug(
                    f"App {app_id}: insufficient releases with reviews ({releases_with_reviews} < {MIN_RELEASES_WITH_REVIEWS})")
                continue

            filtered_apps[app_id] = (releases, stats)
            logger.info(f"App {app_id}: {len(releases)} releases, {releases_with_reviews} with reviews")

        return filtered_apps

    def save_filtered_data(self, filtered_apps: Dict) -> None:
        logger.info("Starting data save process...")

        # Step 1: Save individual app files
        logger.info("Step 1: Saving individual app files...")
        app_files_dir = self.output_dir / "individual_apps"
        app_files_dir.mkdir(exist_ok=True)

        all_releases = []
        saved_count = 0

        for app_id, (releases, stats) in filtered_apps.items():
            try:
                app_file = app_files_dir / f"{app_id}.json"
                app_data = {
                    'app_id': app_id,
                    'statistics': stats,
                    'releases': releases
                }

                with open(app_file, 'w', encoding='utf-8') as f:
                    json.dump(app_data, f, indent=2, default=str)

                all_releases.extend(releases)
                saved_count += 1

                if saved_count % 50 == 0:
                    logger.info(f"  Saved {saved_count}/{len(filtered_apps)} app files")

            except Exception as e:
                logger.error(f"Error saving app {app_id}: {e}")

        logger.info(f"Step 1 complete: Saved {saved_count} app files")

        # Step 2: Create summary CSV
        logger.info("Step 2: Creating summary CSV...")
        try:
            releases_df = pd.DataFrame([
                {
                    'app_id': r['app_id'],
                    'release_id': r['release_id'],
                    'name': r['name'],
                    'date': r['date'],
                    'review_count': r['review_count'],
                    'sequence_number': r['sequence_number'],
                    'is_first_release': r['is_first_release'],
                    'is_last_release': r['is_last_release']
                }
                for r in all_releases
            ])

            releases_csv = self.output_dir / "filtered_releases_summary.csv"
            releases_df.to_csv(releases_csv, index=False)
            logger.info(f"Step 2 complete: Summary CSV saved with {len(releases_df)} rows")

        except Exception as e:
            logger.error(f"Error creating CSV: {e}")

        # Step 3: Save complete data (SKIP - too large)
        logger.info("Step 3: Skipping complete data file (too large for single JSON)")

        logger.info("Data saving completed!")
        logger.info(f"   Individual apps: {app_files_dir}")
        logger.info(f"   Summary CSV: {releases_csv}")

    def generate_statistics_report(self, filtered_apps: Dict) -> Dict:
        logger.info("Generating statistics report...")

        all_releases = []
        for releases, _ in filtered_apps.values():
            all_releases.extend(releases)

        releases_per_app = [len(releases) for releases, _ in filtered_apps.values()]
        review_counts = [r['review_count'] for r in all_releases]

        dates = [r['date'] for r in all_releases]
        date_range = (min(dates), max(dates)) if dates else (None, None)

        report = {
            'filtering_results': {
                'apps_before_filtering': self.stats['apps_processed'],
                'apps_after_filtering': len(filtered_apps),
                'retention_rate': len(filtered_apps) / max(self.stats['apps_processed'], 1),
                'releases_before_filtering': self.stats['total_releases'],
                'releases_after_filtering': len(all_releases),
                'releases_retention_rate': len(all_releases) / max(self.stats['total_releases'], 1)
            },
            'app_statistics': {
                'total_apps': len(filtered_apps),
                'releases_per_app': {
                    'mean': pd.Series(releases_per_app).mean(),
                    'median': pd.Series(releases_per_app).median(),
                    'min': min(releases_per_app) if releases_per_app else 0,
                    'max': max(releases_per_app) if releases_per_app else 0,
                    'std': pd.Series(releases_per_app).std()
                }
            },
            'release_statistics': {
                'total_releases': len(all_releases),
                'releases_with_reviews': sum(1 for r in all_releases if r['review_count'] > 0),
                'reviews_per_release': {
                    'mean': pd.Series(review_counts).mean(),
                    'median': pd.Series(review_counts).median(),
                    'min': min(review_counts) if review_counts else 0,
                    'max': max(review_counts) if review_counts else 0,
                    'std': pd.Series(review_counts).std()
                }
            },
            'temporal_coverage': {
                'date_range': date_range,
                'total_days': (date_range[1] - date_range[0]).days if date_range[0] else 0
            }
        }

        logger.info("Statistics report generated successfully")
        return report

    def run_filtering_pipeline(self) -> Dict:
        logger.info("Starting PRIMA release data filtering pipeline")

        all_apps_data = {}

        json_files = list(self.input_dir.glob("*.json"))
        self.stats['total_files'] = len(json_files)

        logger.info(f"Found {len(json_files)} JSON files to process")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Directory exists: {self.input_dir.exists()}")

        processed_count = 0
        skipped_empty = 0
        skipped_invalid = 0
        skipped_no_dates = 0

        for i, file_path in enumerate(json_files):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(json_files)} files processed")

            app_id = get_app_id_from_filename(file_path.name)

            app_data = self.load_release_data(file_path)
            if not app_data:
                if file_path.stat().st_size < 10:
                    skipped_empty += 1
                else:
                    skipped_invalid += 1
                continue

            releases, stats = self.process_app_releases(app_data, app_id)

            if releases:
                all_apps_data[app_id] = (releases, stats)
                self.stats['apps_processed'] += 1
                self.stats['total_releases'] += stats['total_releases']
                self.stats['releases_with_reviews'] += stats['releases_with_reviews']
                processed_count += 1
            else:
                skipped_no_dates += 1

        logger.info(f"Processing complete:")
        logger.info(f"  Successfully processed: {processed_count}")
        logger.info(f"  Skipped (empty files): {skipped_empty}")
        logger.info(f"  Skipped (invalid format): {skipped_invalid}")
        logger.info(f"  Skipped (no valid dates): {skipped_no_dates}")
        logger.info(f"Processed {len(all_apps_data)} apps with valid releases")

        logger.info("Starting app filtering by criteria...")
        filtered_apps = self.filter_apps_by_criteria(all_apps_data)
        self.stats['apps_kept'] = len(filtered_apps)
        self.stats['final_releases'] = sum(len(releases) for releases, _ in filtered_apps.values())

        logger.info(f"After filtering criteria:")
        logger.info(f"  Apps kept: {len(filtered_apps)}")
        logger.info(f"  Total releases in kept apps: {self.stats['final_releases']}")

        logger.info("Starting data save...")
        self.save_filtered_data(filtered_apps)

        logger.info("Generating final report...")
        report = self.generate_statistics_report(filtered_apps)

        logger.info("Saving report...")
        report_file = self.output_dir / "filtering_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_file}")

        logger.info("Printing summary...")
        self.print_summary(report)

        logger.info("Filtering pipeline completed successfully")
        return filtered_apps

    def print_summary(self, report: Dict) -> None:
        print("\n" + "=" * 60)
        print("PRIMA RELEASE DATA FILTERING SUMMARY")
        print("=" * 60)

        filtering = report['filtering_results']
        apps = report['app_statistics']
        releases = report['release_statistics']
        temporal = report['temporal_coverage']

        print(f"\nFILTERING RESULTS:")
        print(f"   Apps: {filtering['apps_before_filtering']} -> {filtering['apps_after_filtering']} "
              f"({filtering['retention_rate']:.1%} retained)")
        print(f"   Releases: {filtering['releases_before_filtering']} -> {filtering['releases_after_filtering']} "
              f"({filtering['releases_retention_rate']:.1%} retained)")

        print(f"\nAPP STATISTICS:")
        print(f"   Total apps: {apps['total_apps']}")
        print(f"   Releases per app: mean={apps['releases_per_app']['mean']:.1f}, "
              f"median={apps['releases_per_app']['median']:.1f}, "
              f"range=[{apps['releases_per_app']['min']}-{apps['releases_per_app']['max']}]")

        print(f"\nRELEASE STATISTICS:")
        print(f"   Total releases: {releases['total_releases']}")
        print(f"   Releases with reviews: {releases['releases_with_reviews']} "
              f"({releases['releases_with_reviews'] / releases['total_releases']:.1%})")
        print(f"   Reviews per release: mean={releases['reviews_per_release']['mean']:.1f}, "
              f"median={releases['reviews_per_release']['median']:.1f}")

        print(f"\nTEMPORAL COVERAGE:")
        if temporal['date_range'][0]:
            print(f"   Date range: {temporal['date_range'][0].strftime('%Y-%m-%d')} to "
                  f"{temporal['date_range'][1].strftime('%Y-%m-%d')}")
            print(f"   Total span: {temporal['total_days']} days "
                  f"({temporal['total_days'] / 365:.1f} years)")

        print("\n" + "=" * 60)


def main():
    filter_processor = ReleaseFilter()
    filtered_apps = filter_processor.run_filtering_pipeline()

    print(f"\nFiltering complete! {len(filtered_apps)} apps ready for feature extraction.")
    print(f"Results saved in: {FILTERED_RELEASES_DIR}")

    return filtered_apps


if __name__ == "__main__":
    main()