import pandas as pd
import pgeocode
import os
import re

def process_spatial_analysis_efficiently(file_path, country='GB', chunk_size=20000):
    if not os.path.exists(file_path):
        print(f"Error: File not found.")
        return None

    print(f"Starting memory-efficient analysis (Outcode Level)...")
    agg_results = []
    total_ai_overall = 0
    total_firms_overall = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # NEW FIX: Strips brackets [ and quotes ' from the raw CSV data
        chunk['postcodes'] = chunk['postcodes'].astype(str).str.replace(r"[\[\'\"]", "", regex=True).str.strip().str.split().str[0].str.upper()
        
        total_ai_overall += chunk['is_ai_related'].sum()
        total_firms_overall += len(chunk)

        summary = chunk.groupby('postcodes').agg(
            local_total=('domains', 'count'),
            local_ai=('is_ai_related', 'sum')
        ).reset_index()
        agg_results.append(summary)
        print(f"Processed {total_firms_overall} rows...", end='\r')

    final_df = pd.concat(agg_results).groupby('postcodes').sum().reset_index()
    national_density = total_ai_overall / total_firms_overall
    final_df['local_density'] = final_df['local_ai'] / final_df['local_total']
    final_df['LQ'] = final_df['local_density'] / (national_density + 1e-9)

    print(f"\nNational AI Density: {national_density:.4%}")

    print("\nGeocoding unique outcodes...")
    nomi = pgeocode.Nominatim(country)
    # The geocoder will now receive "BS1" instead of "['BS1"
    geo = nomi.query_postal_code(final_df['postcodes'].tolist())
    
    final_df['latitude'] = geo['latitude'].values
    final_df['longitude'] = geo['longitude'].values
    
    # Check how many were successfully geocoded
    geocoded_count = final_df['latitude'].notna().sum()
    print(f"Successfully geocoded {geocoded_count} out of {len(final_df)} areas.")

    # PHASE 4: FILTERING
    # We use LQ >= 1.0 to ensure all "above average" clusters appear in Tableau
    hotspots = final_df[(final_df['local_ai'] >= 1) & (final_df['LQ'] >= 1.0)].copy()
    
    return hotspots

if __name__ == "__main__":
    input_csv = r'D:\Dissertation project\Model\Code\model_outputs\complete_dataset.csv'
    ai_hotspots = process_spatial_analysis_efficiently(input_csv)
    
    if ai_hotspots is not None:
        ai_hotspots = ai_hotspots.sort_values(by='LQ', ascending=False)
        ai_hotspots.to_csv('ai_hotspots_for_tableau.csv', index=False)
        print(f"\nSuccess! Found {len(ai_hotspots)} AI hotspots.")