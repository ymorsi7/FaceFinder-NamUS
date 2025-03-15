# NamUs Scraper Enhancements

I've updated the original NamUs Scraper repository by changing the method, and adding some arguments so that the user can have more freedom on how they scrape. This is mainly because of the amount of data used in this project.

## Changes Made:

### `process-faces.py` Changes:
* Complete rewrite with different face extraction method that processes images individually instead of batch pickle storage
* Argparse implemented (--limit, --input, --model)
* Added 20% padding around the detected faces
* Consistent naming format for the output files using the NamUS ID
  * This is done because in our pipeline, we use this ID to match the faces to the correct person
* Implemented automatic NamUs ID extraction from filepath / filename
* Implemented skipping for duplicates

### `scrape-data.py` Changes:
* Argparse implemented (--limit)

### `scrape-files.py` Changes:
* Argparse implemented (--limit)

<br>
<hr>
<br>

The following is the original `README.md` from the NamUs Scraper repository, where the code was forked from.

> # NamUs Scraper
> Python scraper for collecting metadata and files for Missing-, Unidentified-, and Unclaimed-person cases from the [National Missing and Unidentified Persons System (NamUs)](https://www.namus.gov) organization. The scraper uses APIs used for internal purposes at NamUs and may therefore change at any point.

> To work around the 10.000 case search limit, cases are found by searching on a per-state basis. This may miss some cases if they are entered incorrectly! Compare result counts with the ones available on the site. 

> ⚠️ This requests a large amount of data. Please run it responsibly!

> ## Installation
> ```
> sudo pip3 install requests
> sudo pip3 install grequests
> sudo pip3 install face_recognition
> ```

> ## Scraping
> ```
> python3 scrape-data.py   # Downloads all metadata related to the cases.
> python3 scrape-files.py  # Downloads all files related to the scraped cases.
> python3 process-faces.py # Extracts faces from the downloaded images.
> ```