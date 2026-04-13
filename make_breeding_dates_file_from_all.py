import csv
import sys
import itertools
from pathlib import Path

def process_csv(input_file, output_file, process_row_func):
    """
    Read a CSV file and output rows based on custom logic.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        process_row_func (callable): Function that takes a row dict and returns a list of output rows
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            print(f"Processing input file: {input_file}")
            # Use csv.reader to properly handle multi-line cells
            temp_reader = csv.reader(infile)
            # Skip first 2 rows (rows 1-2), headers are on row 3
            next(temp_reader)
            next(temp_reader)
            # Now create DictReader from the remaining rows
            reader = csv.DictReader(temp_reader)  # type: ignore
            headers = next(temp_reader)  # Row 3 is the headers
            
            # Create list of remaining rows
            data_rows = list(temp_reader)
            
            # Create a simple reader that can iterate over the data
            class SimpleReader:
                def __init__(self, headers, rows):
                    self.fieldnames = headers
                    self.rows = rows
                    self.index = 0
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    if self.index >= len(self.rows):
                        raise StopIteration
                    row_data = self.rows[self.index]
                    self.index += 1
                    return {k: v for k, v in zip(self.fieldnames, row_data)}
            
            reader = SimpleReader(headers, data_rows)
            
            print(f"First few fieldnames: {reader.fieldnames[:5]}")
            print(f"Total fieldnames: {len(reader.fieldnames)}")
            
            row_count = 0
            output_count = 0
            all_output_rows = []
            output_fieldnames = None
            
            for row in reader:
                row_count += 1
                # Process the row and get output rows
                output_rows = process_row_func(row)
                
                # Collect output rows and determine fieldnames from first row
                for output_row in output_rows:
                    all_output_rows.append(output_row)
                    if output_fieldnames is None:
                        output_fieldnames = list(output_row.keys())
                    output_count += 1
            
            # Write output file
            if output_fieldnames and all_output_rows:
                print(f"Writing output file: {output_file}")
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                    writer.writeheader()
                    writer.writerows(all_output_rows)
                
                print(f"Processed {row_count} input rows")
                print(f"Generated {output_count} output rows")
                print(f"Output written to: {output_file}")

                import pandas as pd
                from pathlib import Path

                df = pd.DataFrame(all_output_rows)
                breeding_type_counts = df["Breeding Type"].value_counts()
                print(breeding_type_counts)

                output_path = Path("results.txt")
                with output_path.open("w", encoding="utf-8") as f:
                    f.write("Breeding Type Summary\n")
                    f.write("=====================\n\n")
                    f.write(f"{breeding_type_counts}")
            else:
                print(f"Processed {row_count} input rows")
                print(f"No output rows generated (all rows may have been skipped)")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


def example_process_row(row):
    """
        try:
            with open(input_file, 'r', encoding='utf-8') as infile:
                # Create a temporary csv.reader to skip the first 2 rows
                temp_reader = csv.reader(infile)
                next(temp_reader)  # Skip row 1
                next(temp_reader)  # Skip row 2
                # Convert remaining rows back to strings for DictReader
                remaining_rows = ([''.join(f for f in row) if isinstance(row, list) else row for row in temp_reader])
            
                # Get headers from the next row (row 3)
                reader_iter = iter(remaining_rows)
                headers = next(reader_iter)
                if isinstance(headers, list):
                    headers = headers
            
                # Create DictReader with headers and remaining data
                reader = csv.DictReader(reader_iter, fieldnames=headers)
    Returns:
        list: List of dictionaries (rows to output)
    """
    output_rows = []
    
    # Skip rows where "Skip Site" = Y
    if row["Skip Site"] == "Y":
        return output_rows
    
    # Extract base site information
    site_id = row.get('Site ID', '')
    group = row.get('Group', '')
    pretty_site_name = row.get('Pretty Site Name', '')
    site_name = row.get('Name')
    comment = row.get('Comment for Skip Site')
    breeding_type = row.get('Breeding Type', '')
    complex_types = row.get('Complex Types', '')
    first_rec = row.get('First Recording', '')
    last_rec = row.get('Last Recording', '')
    substrate = row.get('Substrate')
    colony_size = row.get('Approx Colony Size')

    # Define pulse outcomes to process
    pulses = ['p1', 'p2', 'p3', 'p4']
    outcome_columns = ['Outcome']
    date_columns = ['mcstart', 'incstart', 'hatch', 'fledgestart', 'fledgedisp', 'abandon']
    
    # Process each pulse
    for pulse in pulses:
        outcome_col = pulse + 'Outcome'
        outcome = row.get(outcome_col, 'n/a').strip()
        
        # Skip if outcome is "n/a" or empty
        if outcome.lower() == 'n/a' or not outcome:
            continue
        
        # Create output row with base columns
        output_row = {
            'Site ID': site_id,
            'Group': group,
            'Name' : site_name,
            'Pretty Name': pretty_site_name,
            'Comment' : comment,
            'Deployment Start': first_rec,
            'Deployment End' : last_rec,
            'Breeding Type': breeding_type,
            'Complex Types': complex_types,
            'Pulse Name': pretty_site_name + ' ' + pulse,
            'Outcome': outcome,
            'Substrate' : substrate, 
            'Colony Size' : colony_size,     
        }
        
        # Add pulse-specific date columns
        for date_col in date_columns:
            col_name = pulse + date_col
            val = row.get(col_name, '')
            # If the hatch value starts with "before", normalize to 'pre'
            # If the Outcome is Partially Abandoned, the actual abandon value
            # should be treated as 'ND' in the output
            if date_col == 'abandon' and isinstance(outcome, str) and outcome.strip().lower() == 'partially abandoned':
                output_row[date_col] = 'ND'
            elif date_col == 'hatch' and isinstance(val, str) and val.strip().lower().startswith('before'):
                output_row[date_col] = 'pre'
            else:
                output_row[date_col] = val

        # Add 'partial abandon' column:
        # If the pulse Outcome is 'Partially Abandoned', the corresponding pNabandon
        # will contain a trailing 'P' or 'p' — write it without that trailing char.
        # Otherwise write 'ND'.
        partial_val = 'ND'
        try:
            if isinstance(outcome, str) and outcome.strip().lower() == 'partially abandoned':
                raw_abandon = row.get(pulse + 'abandon', '')
                if isinstance(raw_abandon, str) and raw_abandon:
                    raw_abandon = raw_abandon.strip()
                    if raw_abandon.endswith(('P', 'p')):
                        partial_val = raw_abandon[:-1]
                    else:
                        partial_val = raw_abandon
        except Exception:
            partial_val = 'ND'

        output_row['partial abandon'] = partial_val
        
        output_rows.append(output_row)
    
    return output_rows

def make_breeding_dates_file():
    # Configuration
    INPUT_FILE = r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"
    OUTPUT_FILE = 'breeding dates.csv'
    
    # Run the processor
    process_csv(INPUT_FILE, OUTPUT_FILE, example_process_row)



if __name__ == '__main__':
    make_breeding_dates_file()
