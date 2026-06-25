
import pandas as pd
import html
import os

# data cleaning 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "anime-datas.csv")

df = pd.read_csv(data_path)

print(df.columns.tolist())

print("Shape",df.shape)
print("\n First 5 rows :")
print(df.head(5))
print("\n Column types:")
df.info()

print(df[["Score", "Rating"]].head())

df = df.drop(columns=["English name" ,"Other name" ,"Aired","Premiered","Status","Producers","Licensors","Studios","Source","Duration","Rank","Popularity","Favorites","Scored By","Rating"])

df["Episodes"] = pd.to_numeric(df["Episodes"], errors="coerce")
df["Score"]   = pd.to_numeric(df["Score"],   errors="coerce")
df["Members"]  = pd.to_numeric(df["Members"],  errors="coerce")

print("\nMissing values per column:")
print(df.isna().sum())

df['Display_name'] = df['Name']

median_val = df['Episodes'].median()
df['Episodes'] = df['Episodes'].fillna(median_val)

median_val = df['Score'].median()
df['Score'] = df['Score'].fillna(median_val)

print(f"\nDuplicates found: {df.duplicated().sum()}")
df.drop_duplicates(subset=["anime_id"], inplace=True)

df.dropna(subset = ["Name","Genres","Synopsis","Type"] ,inplace=True)

df["Name"] = df["Name"].apply(html.unescape)
df["Name"] = df["Name"].str.replace(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', regex=True)
df["Name"] = df["Name"].str.strip().str.lower()

df["Genres"] = df["Genres"].str.replace("unknown", "" , case = False)
df["Genres"] = df["Genres"].fillna("")
df['Genres'] = df['Genres'].str.lower().str.strip()

df["Synopsis"] = df["Synopsis"].str.replace("unknown", "", case=False)
df["Synopsis"] = df["Synopsis"].fillna("")
df = df[df["Synopsis"].str.strip() != ""]

df['Type'] = df['Type'].str.lower().str.strip()

empty_before = df[df["Name"] == ""].shape[0]
print(f"\nEmpty names before fix: {empty_before}")


df.loc[df["Name"] == "", "Name"] = (
    df.loc[df["Name"] == "", "Display_name"]
    .apply(html.unescape)
    .str.strip()
    .str.lower()
)

empty_after = df[df["Name"] == ""].shape[0]
print(f"Empty names after fix:  {empty_after}")

df = df.reset_index(drop=True)

df = df.rename(columns={"Image URL": "Image_URL"})
df["combined_features"] = df["Genres"] + " " + df["Synopsis"]

df = df.drop(columns = ["Display_name"])

print(df["combined_features"].head(5))

print("\nFinal shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isna().sum())
print("\nSample:")
print(df.head())


output_path = os.path.join(BASE_DIR, "data", "anime-data-cleaned.csv")
df.to_csv(output_path, index=False)




