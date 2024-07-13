import pandas as pd

# Read the CSV files
train_df = pd.read_csv('data/dirty_walmart_amazon/deep_matcher/valid.csv')
tableA_df = pd.read_csv('data/dirty_walmart_amazon/deep_matcher/tableA.csv')
tableB_df = pd.read_csv('data/dirty_walmart_amazon/deep_matcher/tableB.csv')

# Function to get the row from tableA and tableB based on IDs in train file
def get_matching_rows(train_row, tableA_df, tableB_df):
    ltable_id = train_row['ltable_id']
    rtable_id = train_row['rtable_id']
    
    # Find the corresponding rows in tableA and tableB
    tableA_row = tableA_df[tableA_df['id'] == ltable_id]
    tableB_row = tableB_df[tableB_df['id'] == rtable_id]
    
    return tableA_row, tableB_row


def apply_template(row):
    #base : concat
    cols = ['title','category','brand','modelno','price']
    text = ""
    for c in cols:
        if not row[c].isna().values[0]:
            text += str(row[c].values[0])
    
    return text

    #use template
    # text = ''
    # if not row['title'].isna().values[0]:
    #     text += f'Introducing the {row['title'].values[0]}'

    # if not row['category'].isna().values[0] and not row['brand'].values[0]:
    #     text += f", a product in the {row['category'].values[0]} category by {row['brand'].values[0]}."
    # elif not row['category'].isna().values[0]:
    #     text += f", in the {row['category'].values[0]} category."
    # elif not row['brand'].isna().values[0]:
    #     text += f" by {row['brand'].values[0]}."
        
    # if not row['modelno'].isna().values[0] and not row['price'].isna().values[0]:
    #     text += f" This model, {row['modelno'].values[0]}, is available for {row['price'].values[0]}."
    # elif not row['modelno'].isna().values[0]:
    #     text += f" Its model is {row['modelno'].values[0]}."
    # elif not row['price'].isna().values[0]:
    #     text += f" This product is available for {row['price'].values[0]}."

    # return text


# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['idx', 'text_left', 'text_right', 'label'])

# Iterate through each row in the train file and store the corresponding rows in tableA and tableB
for idx, train_row in train_df.iterrows():
    tableA_row, tableB_row = get_matching_rows(train_row, tableA_df, tableB_df)
    
    # Assuming 'text' is the column of interest in tableA and tableB
    text_left = apply_template(tableA_row)
    text_right = apply_template(tableB_row)
    label = train_row['label']
    
    # Create a new DataFrame for the current row
    new_row = pd.DataFrame([{
        'idx': idx,
        'text_left': text_left,
        'text_right': text_right,
        'label': label
    }])
    
    # Append the new row to the result DataFrame using pd.concat
    result_df = pd.concat([result_df, new_row], ignore_index=True)


# Write the result DataFrame to a .tsv file
result_df.to_csv('data/dirty_walmart_amazon/dev.tsv', sep='\t', index=False)

# Introducing the [Title], a product in the [Category] category by [Brand]. This model, [Model No.], is available for  [Price].