"""
Selenium script for scraping The Valley of the Shadow
(https://valley.newamericanhistory.org) Augusta County letters
(https://valley.newamericanhistory.org/search/letters/results?county=augusta).
Accessed 2024-12-17 & 2023-12-18.

The output is two directories: metadata/ and content/

- metadata/ contains JSON files containing each letter's metadata:
    - title: The letter's title from the webpage
    - summary: The letter's summary from the webpage
    - keywords: The letters keywords from the webpage
    - source_file: The URL pointing to the raw XML sourcefile for the webpage
    - content_file: The local relative path to the HTML contents of the webpage

- content/ contains HTML files containing each letter's contents markup,
    excluding any meta tags and UI related markup.

The files are named using the format p{p}_{l}.{json | html}
where p = search results page number zero padded to three digits
where l = letter number within search results page zero padded to three digits
"""
import json
import os
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Path to the WebDriver executable
driver_path = "/Applications/chromedriver"

# Base URL for the search results
base_url = "https://valley.newamericanhistory.org/search/letters/results?county=augusta&page={}"

# The first and last page of search results, inclusive
first_page = 1
last_page = 43

# Initialize the WebDriver
service = Service(driver_path)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# Directories for storing content and metadata
os.makedirs("letters/content", exist_ok=True)
os.makedirs("letters/metadata", exist_ok=True)


def extract_letter_links(driver):
    """
    Extracts all letter links from the current search results page.

    :param driver: The webdriver to be used by Selenium.
    :type driver: selenium.webdriver
    :rtype list: List of strings representing links to letter pages.
    """
    # Wait for the card list to be present
    card_list = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "search-results__card-list"))
    )

    # Find all letter links
    letter_links = card_list.find_elements(By.TAG_NAME, "a")
    return [link.get_attribute("href") for link in letter_links]


def extract_letter_content(driver):
    """
    Extracts letter content and metadata from the page.

    :param driver: The webdriver to be used by Selenium.
    :type driver: selenium.webdriver
    :rtype tuple: Tuple of strings where item at 0 is stringified HTML
    content of letter page, and item 1 is stringified JSON metadata of letter
    page.
    """
    try:
        # Wait for the main content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "source__content"))
        )

        # Extract metadata
        metadata = {
            "title": driver.find_element(By.CLASS_NAME, "page__title").text,
            "summary": driver.find_element(By.CLASS_NAME, "source__summary").text
            if len(driver.find_elements(By.CLASS_NAME, "source__summary")) > 0
            else "",
            "keywords": driver.find_element(
                By.CLASS_NAME, "source__footer-field__value"
            ).text.strip()
            if len(driver.find_elements(By.CLASS_NAME, "source__footer-field__value"))
            > 0
            else "",
            "source_file": driver.find_element(
                By.CLASS_NAME, "source__footer-field__value--ellipsis"
            ).text
            if len(
                driver.find_elements(
                    By.CLASS_NAME, "source__footer-field__value--ellipsis"
                )
            )
            > 0
            else "",
        }

        # Extract only the letter HTML content
        letter_element = driver.find_element(By.ID, "tab-spelling-original")
        html_content = letter_element.get_attribute("innerHTML")

        return metadata, html_content

    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        return None, None


def process_page(page_number):
    """
    Processes a single page of search results.

    :param page_number: An integer indicating the search results page number
    to scrape from
    https://valley.newamericanhistory.org/search/letters/results?county=augusta
    from first_page to last_page, inclusive (1, 43).
    :type page_number: int
    :rtype bool: Boolean indicating if page results were fetched and saved to
    disc succesfully or not.
    """
    try:
        # Create and fetch the specific results page URL
        url = base_url.format(page_number)
        driver.get(url)

        # Extract letter links
        letter_links = extract_letter_links(driver)

        print(f"Found {len(letter_links)} letters on page {page_number}")

        for index, link in enumerate(letter_links):
            try:
                # Fetch the letter markup and metadata
                driver.get(link)
                metadata, content = extract_letter_content(driver)

                if metadata and content:
                    # Generate unique base_filename
                    base_filename = f"p{page_number:03d}_{index:03d}"

                    # Save letter markup
                    with open(
                        f"letters/content/{base_filename}.html", "w", encoding="utf-8"
                    ) as f:
                        f.write(str(content))

                    # Save metadata with reference to content file
                    metadata["content_file"] = f"content/{base_filename}.html"
                    with open(
                        f"letters/metadata/{base_filename}.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(metadata, f, indent=2)

                    print(f"Saved letter {base_filename}: {metadata['title'][:50]}...")

                # Wait a short amount of time to now overwhelm the provider's server
                time.sleep(2)

            except Exception as e:
                # Echo out to user if anything went wrong
                print(f"Error processing letter {link}: {str(e)}")
                continue

        # Success
        return True

    except Exception as e:
        # Echo out to user if anything went wrong
        print(f"Error processing page {page_number}: {str(e)}")
        return False


try:
    # Loop through every page of search results and scrape
    for page in range(first_page, last_page + 1):
        print(f"\nProcessing page {page}")
        success = process_page(page)
        if not success:
            # Echo out to user if anything went wrong
            print(f"Failed to process page {page}, continuing to next page")
        # Wait a short amount of time to now overwhelm the provider's server
        time.sleep(3)

finally:
    # Clean up resources
    driver.quit()