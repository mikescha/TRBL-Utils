from make_acoustic_ratios import make_ratios
from make_breeding_dates_file_from_all import make_breeding_dates_file
from make_recordings_per_day import build_recordings_per_day_file
from make_recordings_per_day_per_hour import build_recordings_per_day_per_hour_file


def main() -> None:
    print("Making support files...")
    
    print("Starting with expanding the All file to breeding dates per pulse...")
    make_breeding_dates_file()

    print("Compressing recordings into a per-day table...")
    build_recordings_per_day_file()

    print("Compressing recordings into a per-day and per-hour table...")
    build_recordings_per_day_per_hour_file()

    print("Generating ratios of nestling to female calls...")
    make_ratios()
   
if __name__ == "__main__":
    main()
