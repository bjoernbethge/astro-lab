"""
CLI commands for data information and inspection.
"""

from ..data.info import SurveyInfo


def show_info(args) -> int:
    """Show information about survey(s)."""
    info = SurveyInfo()

    # If no survey specified or 'all', show all surveys
    if not hasattr(args, "survey") or args.survey == "all":
        info.print_all_surveys_status()
        return 0

    # Show specific survey info
    survey = args.survey

    # Check if survey exists
    available_surveys = info.list_available_surveys()
    if survey not in available_surveys:
        print(f"\n❌ Error: Unknown survey '{survey}'")
        print(f"Available surveys: {', '.join(available_surveys)}")
        return 1

    # Show basic info
    if not args.columns and not args.validate:
        info.print_survey_summary(survey)

        # Show sample if requested
        if args.sample > 0:
            data = info.inspect_survey_data(survey, sample_size=args.sample)
            if "sample" in data and data["sample"]:
                print(f"\n📋 Sample Data ({args.sample} rows):")
                print("-" * 80)
                import json

                print(json.dumps(data["sample"], indent=2, default=str))

    # Show columns if requested
    if args.columns:
        data = info.inspect_survey_data(survey, sample_size=0)

        if "error" in data:
            print(f"\n❌ Error: {data['error']}")
            return 1

        print(f"\n📊 Columns for {survey.upper()}:")
        print("=" * 100)
        print(
            f"{'Column':<30} {'Type':<15} {'Nulls %':<10} {'Min':<12} {'Max':<12} {'Mean':<12}"
        )
        print("-" * 100)

        for col in data["columns"]:
            if col in data["column_info"]:
                col_info = data["column_info"][col]

                # Format numeric values
                if "mean" in col_info:
                    min_val = f"{col_info['min']:.2f}"
                    max_val = f"{col_info['max']:.2f}"
                    mean_val = f"{col_info['mean']:.2f}"
                else:
                    min_val = max_val = mean_val = "N/A"

                print(
                    f"{col:<30} {col_info['dtype']:<15} "
                    f"{col_info['null_percentage']:<10.1f} "
                    f"{min_val:<12} {max_val:<12} {mean_val:<12}"
                )

    # Run validation if requested
    if args.validate:
        data = info.inspect_survey_data(survey, sample_size=0)

        if "error" in data:
            print(f"\n❌ Error: {data['error']}")
            return 1

        print(f"\n🔍 Validating {survey.upper()} data...")
        print("=" * 60)

        validation = data["validation"]

        if validation["is_valid"]:
            print("\n✅ Data validation: PASSED")
            print("All data quality checks passed successfully.")
        else:
            print("\n⚠️  Data validation: ISSUES FOUND")
            print("\nIssues:")
            for i, issue in enumerate(validation["issues"], 1):
                print(f"  {i}. {issue}")

    return 0


def main(args):
    """Main function for CLI info command (only new argument structure)."""
    return show_info(args)
