import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def filter_and_save_releases_with_reviews(input_folder, output_folder):
    """
    Filter out releases with 0 reviews and save each app as a separate JSON file,
    but only for apps that have at least one release with reviews.
    
    Args:
        input_folder (str): Path to the folder containing release JSON files
        output_folder (str): Path to the folder where to save the filtered JSON files
    """
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing release files from: {input_path}")
    print("Filtering out releases with 0 reviews...")
    
    processed_files = 0
    saved_files = 0
    total_releases = 0
    releases_with_reviews = 0
    
    for json_file in input_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter releases to only include those with reviews
            filtered_releases = []
            for release in data:
                reviews = release.get('google_play_reviews', [])
                if len(reviews) > 0:  # Only include releases with at least 1 review
                    filtered_releases.append(release)
                    releases_with_reviews += 1
                total_releases += 1
            
            # Only save the file if there are releases with reviews
            if filtered_releases:
                # Save as separate file with same name
                output_file = output_path / json_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_releases, f, indent=2, ensure_ascii=False)
                saved_files += 1
                
            processed_files += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\n📊 FILTERING RESULTS:")
    print(f"Total files processed: {processed_files}")
    print(f"Files saved (with reviews): {saved_files}")
    print(f"Files skipped (no reviews): {processed_files - saved_files}")
    print(f"Total releases before filtering: {total_releases}")
    print(f"Releases with reviews (kept): {releases_with_reviews}")
    print(f"Releases with 0 reviews (filtered out): {total_releases - releases_with_reviews}")
    print(f"Filtered files saved to: {output_path}")
    
    return saved_files

def analyze_json_files(input_folder=None, output_folder=None):
    """
    Analyze all JSON files in the specified directory
    and generate plots and statistics about releases and their embedded reviews.
    
    Args:
        input_folder (str): Path to the folder containing release JSON files
        output_folder (str): Path to the folder where to save outputs
    """
    
    # Use default paths if not provided
    if input_folder is None:
        base_path = Path("data/input/DATAR/release_related")
        input_folder = base_path / "all_jsons"
    else:
        input_folder = Path(input_folder)
    
    if output_folder is None:
        output_folder = Path("data/output")
    else:
        output_folder = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Data storage
    app_release_counts = []
    app_review_counts = []
    release_review_counts = []
    
    # Process release files
    print("Processing release files...")
    for json_file in input_folder.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count releases for this app
            num_releases = len(data)
            app_name = json_file.stem  # Remove .json extension
            app_release_counts.append((app_name, num_releases))
            
            # Extract review data from google_play_reviews property
            total_reviews_for_app = 0
            for release in data:
                reviews = release.get('google_play_reviews', [])
                num_reviews = len(reviews)
                release_review_counts.append(num_reviews)
                total_reviews_for_app += num_reviews
            
            app_review_counts.append((app_name, total_reviews_for_app))
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create plots
    create_plots(app_release_counts, release_review_counts, app_review_counts, output_folder)
    
    # Print statistics
    print_statistics(app_release_counts, release_review_counts, app_review_counts)

def create_plots(app_release_counts, release_review_counts, app_review_counts, output_folder):
    """Create the requested histograms and plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram with number of releases per app
    release_counts = [count for _, count in app_release_counts]
    axes[0, 0].hist(release_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Releases per App')
    axes[0, 0].set_ylabel('Number of Apps')
    axes[0, 0].set_title('Distribution of Releases per App')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram with number of reviews per release
    axes[0, 1].hist(release_review_counts, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Reviews per Release')
    axes[0, 1].set_ylabel('Number of Releases')
    axes[0, 1].set_title('Distribution of Reviews per Release')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot: releases vs reviews per app
    app_release_dict = dict(app_release_counts)
    app_review_dict = dict(app_review_counts)
    
    # Match apps that have both release and review data
    matched_apps = []
    for app_name in set(app_release_dict.keys()) & set(app_review_dict.keys()):
        matched_apps.append((app_release_dict[app_name], app_review_dict[app_name]))
    
    if matched_apps:
        releases, reviews = zip(*matched_apps)
        axes[1, 0].scatter(releases, reviews, alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Number of Releases')
        axes[1, 0].set_ylabel('Number of Reviews')
        axes[1, 0].set_title('Releases vs Reviews per App')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of reviews per release
    axes[1, 1].boxplot(release_review_counts, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 1].set_ylabel('Number of Reviews per Release')
    axes[1, 1].set_title('Distribution of Reviews per Release (Box Plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_folder / "release_analysis_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(app_release_counts, release_review_counts, app_review_counts):
    """Print comprehensive statistics about the data."""
    
    print("\n" + "="*60)
    print("RELEASE AND REVIEW ANALYSIS STATISTICS")
    print("="*60)
    
    # App-level statistics
    release_counts = [count for _, count in app_release_counts]
    review_counts = [count for _, count in app_review_counts]
    
    print(f"\n📊 APP-LEVEL STATISTICS:")
    print(f"Total number of apps with release data: {len(app_release_counts)}")
    print(f"Total number of apps with review data: {len(app_review_counts)}")
    print(f"Average releases per app: {np.mean(release_counts):.2f}")
    print(f"Median releases per app: {np.median(release_counts):.2f}")
    print(f"Average reviews per app: {np.mean(review_counts):.2f}")
    print(f"Median reviews per app: {np.median(review_counts):.2f}")
    
    # Release-level statistics
    print(f"\n📈 RELEASE-LEVEL STATISTICS:")
    print(f"Total number of releases: {len(release_review_counts)}")
    print(f"Average reviews per release: {np.mean(release_review_counts):.2f}")
    print(f"Median reviews per release: {np.median(release_review_counts):.2f}")
    print(f"Releases with 0 reviews: {sum(1 for x in release_review_counts if x == 0)}")
    print(f"Releases with 1+ reviews: {sum(1 for x in release_review_counts if x > 0)}")
    print(f"Releases with 5+ reviews: {sum(1 for x in release_review_counts if x >= 5)}")
    print(f"Releases with 10+ reviews: {sum(1 for x in release_review_counts if x >= 10)}")
    
    # Coverage analysis
    releases_with_reviews = sum(1 for x in release_review_counts if x > 0)
    total_releases = len(release_review_counts)
    coverage_percentage = (releases_with_reviews / total_releases) * 100 if total_releases > 0 else 0
    
    print(f"\n🎯 REVIEW COVERAGE ANALYSIS:")
    print(f"Percentage of releases with reviews: {coverage_percentage:.1f}%")
    print(f"Releases with reviews: {releases_with_reviews}")
    print(f"Total releases: {total_releases}")
    
    # Top apps by releases
    print(f"\n🏆 TOP 10 APPS BY NUMBER OF RELEASES:")
    sorted_releases = sorted(app_release_counts, key=lambda x: x[1], reverse=True)
    for i, (app, count) in enumerate(sorted_releases[:20], 1):
        print(f"{i:2d}. {app}: {count} releases")
    
    # Top apps by reviews
    print(f"\n🏆 TOP 10 APPS BY NUMBER OF REVIEWS:")
    sorted_reviews = sorted(app_review_counts, key=lambda x: x[1], reverse=True)
    for i, (app, count) in enumerate(sorted_reviews[:20], 1):
        print(f"{i:2d}. {app}: {count} reviews")

def main():
    """Main function to handle command line arguments and execute the appropriate function."""
    parser = argparse.ArgumentParser(description='Analyze and filter release data with reviews')
    parser.add_argument('--input-folder', type=str, 
                       default="data/input/DATAR/release_related/all_jsons",
                       help='Path to the folder containing release JSON files')
    parser.add_argument('--output-folder', type=str, 
                       default="data/output",
                       help='Path to the folder where to save outputs')
    parser.add_argument('--filter-only', action='store_true',
                       help='Only filter releases with reviews, skip analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run analysis, skip filtering')
    
    args = parser.parse_args()
    
    if args.filter_only:
        # Only filter and save releases with reviews
        filter_and_save_releases_with_reviews(args.input_folder, args.output_folder)
    elif args.analyze_only:
        # Only run analysis
        analyze_json_files(args.input_folder, args.output_folder)
    else:
        # Run both filtering and analysis
        print("=== FILTERING RELEASES WITH REVIEWS ===")
        filter_and_save_releases_with_reviews(args.input_folder, args.output_folder)
        print("\n=== RUNNING ANALYSIS ===")
        analyze_json_files(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
