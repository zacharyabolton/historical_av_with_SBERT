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

first_page = 1
last_page = 43

# Initialize the WebDriver
service = Service(driver_path)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# Create directories for storing content
os.makedirs("letters/content", exist_ok=True)
os.makedirs("letters/metadata", exist_ok=True)


def extract_letter_links(driver):
    """
    Extracts all letter links from the current search results page.
    """
    # Wait for the card list to be present
    card_list = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "search-results__card-list"))
    )

    # Find all letter links
    letter_links = card_list.find_elements(By.TAG_NAME, "a")
    return [link.get_attribute("href") for link in letter_links]


def extract_letter_content(driver):
    """Extracts letter content and metadata from the page."""
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

        # Extract letter HTML content
        letter_element = driver.find_element(By.ID, "tab-spelling-original")
        html_content = letter_element.get_attribute("innerHTML")

        return metadata, html_content

    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        return None, None


def process_page(page_number):
    """Processes a single page of search results."""
    try:
        url = base_url.format(page_number)
        driver.get(url)

        # Extract letter links
        letter_links = extract_letter_links(driver)

        print(f"Found {len(letter_links)} letters on page {page_number}")

        for index, link in enumerate(letter_links):
            try:
                driver.get(link)
                metadata, content = extract_letter_content(driver)

                if metadata and content:
                    # Generate unique ID for the letter
                    letter_id = f"p{page_number:03d}_{index:03d}"

                    # Save content
                    with open(
                        f"letters/content/{letter_id}.html", "w", encoding="utf-8"
                    ) as f:
                        f.write(str(content))

                    # Save metadata with reference to content file
                    metadata["content_file"] = f"content/{letter_id}.html"
                    with open(
                        f"letters/metadata/{letter_id}.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(metadata, f, indent=2)

                    print(f"Saved letter {letter_id}: {metadata['title'][:50]}...")

                time.sleep(2)  # Polite delay between letters

            except Exception as e:
                print(f"Error processing letter {link}: {str(e)}")
                continue

        return True

    except Exception as e:
        print(f"Error processing page {page_number}: {str(e)}")
        return False


try:
    for page in range(first_page, last_page + 1):
        print(f"\nProcessing page {page}")
        success = process_page(page)
        if not success:
            print(f"Failed to process page {page}, continuing to next page")
        time.sleep(3)  # Polite delay between pages

finally:
    driver.quit()
