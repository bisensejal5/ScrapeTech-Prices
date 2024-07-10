import requests
from bs4 import BeautifulSoup
from colorthief import ColorThief
import matplotlib.colors as mcolors
import re
import io
from flask import Flask, request, render_template_string
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin

app = Flask(__name__)

# HTML templates
index_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Color Extractor</title>
</head>
<body>
    <div style="text-align: center;">
        <h1>Enter a URL to scrape:</h1>
        <form action="/scrape" method="POST">
            <input type="text" name="url" placeholder="http://example.com" required style="width: 300px;">
            <button type="submit">Scrape</button>
        </form>
    </div>
</body>
</html>
"""

result_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Scraping Results</title>
</head>
<body>
    <div style="text-align: center;">
        <h1>Scraping Results</h1>
        <p><strong>Logo URL:</strong> <a href="{{ result['logo_url'] }}">{{ result['logo_url'] }}</a></p>
        <p><strong>Primary Colors:</strong> {{ result['primary_colors'] }}</p>
        <p><strong>Button Colors:</strong> {{ result['button_colors'] }}</p>
        <p><strong>Recommended Button Color:</strong> {{ result['recommended_button_color'] }}</p>
    </div>
    <div style="text-align: center;">
        <a href="/">Go back</a>
    </div>
</body>
</html>
"""


# Helper functions
def get_logo_url(soup, base_url):
    logo = soup.find('img', {'alt': re.compile('logo', re.I)}) or \
           soup.find('img', {'src': re.compile('logo', re.I)})
    if logo and 'src' in logo.attrs:
        return urljoin(base_url, logo['src'])
    return None


def extract_primary_colors(logo_url):
    try:
        response = requests.get(logo_url)
        response.raise_for_status()
        color_thief = ColorThief(io.BytesIO(response.content))
        palette = color_thief.get_palette(color_count=5)
        hex_palette = [mcolors.rgb2hex(color) for color in palette]
        return hex_palette
    except Exception as e:
        print(f"Failed to extract colors from logo: {e}")
        return []


def get_button_colors(soup):
    buttons = soup.find_all('button')
    colors = set()
    for button in buttons:
        style = button.get('style')
        if style:
            match = re.search(r'background-color:\s*(#[0-9a-fA-F]{6}|rgb\(.+?\));', style)
            if match:
                colors.add(match.group(1))
        classes = button.get('class', [])
        for class_name in classes:
            css = soup.find('style', text=re.compile(class_name))
            if css:
                match = re.search(
                    r'\.{}.+?background-color:\s*(#[0-9a-fA-F]{6}|rgb\(.+?\));'.format(re.escape(class_name)),
                    css.string)
                if match:
                    colors.add(match.group(1))
    return list(colors)


def recommend_button_color(primary_colors, button_colors):
    all_colors = primary_colors + button_colors
    contrasting_colors = ['#000000', '#ffffff', '#ff5733', '#28a745']
    for color in contrasting_colors:
        if color not in all_colors:
            return color
    return contrasting_colors[0]


def scrape_website(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    logo_url = get_logo_url(soup, url)
    primary_colors = extract_primary_colors(logo_url) if logo_url else []
    button_colors = get_button_colors(soup)
    recommended_color = recommend_button_color(primary_colors, button_colors)

    return {
        'logo_url': logo_url,
        'primary_colors': primary_colors,
        'button_colors': button_colors,
        'recommended_button_color': recommended_color
    }


# Flask routes
@app.route('/')
def index():
    return render_template_string(index_html)


@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url')
    if not url:
        return "No URL provided", 400

    result = scrape_website(url)
    return render_template_string(result_html, result=result)


if __name__ == '__main__':
    app.run(debug=True)
