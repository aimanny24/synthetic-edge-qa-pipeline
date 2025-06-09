import great_expectations as ge
import json

def validate_data(df):
    df_ge = ge.from_pandas(df)

    df_ge.expect_column_values_to_be_between("Amount", min_value=0, max_value=100000)
    df_ge.expect_column_values_to_not_be_null("Time")
    df_ge.expect_column_values_to_be_of_type("Time", "float64")

    result = df_ge.validate()

    # ✅ Convert to dictionary before saving
    result_dict = result.to_dict()

    with open("data/validation_report.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    return result
