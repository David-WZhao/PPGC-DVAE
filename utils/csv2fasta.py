import csv


def csv_to_fasta(csv_file, fasta_file):
    """
    Converts a CSV file to a FASTA file.

    Parameters:
        csv_file (str): Path to the input CSV file.
        fasta_file (str): Path to the output FASTA file.
    """
    try:
        with open(csv_file, 'r') as csvfile, open(fasta_file, 'w') as fastafile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row if it exists

            for row in reader:
                if len(row) < 2:
                    print(f"Skipping invalid row: {row}")
                    continue

                sequence_id, sequence = row[0], row[1]
                fastafile.write(f">{sequence_id}\n")
                fastafile.write(f"{sequence}\n")

        print(f"FASTA file has been successfully created: {fasta_file}")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
csv_to_fasta("input.csv", "output.fasta")