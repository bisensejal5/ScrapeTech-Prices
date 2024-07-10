# ScrapeTech-Prices
This repository contains Python code for scraping laptop data from Amazon.in, performing exploratory data analysis (EDA), and building predictive models to analyze the relationship between laptop specifications and prices.

#### Key Components:

1. **Data Scraping (`fetch_amazon_data`, `fetch_all_data`):**
   - Utilizes `requests`, `BeautifulSoup` to scrape laptop titles, prices, ratings, and specifications from multiple pages on Amazon.in.
   - Data is stored in a Pandas DataFrame and saved to `amazon_laptops_raw.csv`.

2. **Data Cleaning:**
   - Handles missing values and inconsistencies in the scraped data.
   - Extracts RAM and Storage information from the laptop titles.

3. **Exploratory Data Analysis (EDA):**
   - Visualizes the distribution of laptop prices, and explores relationships between prices and RAM, and prices and storage using `matplotlib` and `seaborn`.
   - Includes a correlation matrix heatmap to understand feature relationships.

4. **Machine Learning Models:**
   - **Linear Regression:** Predicts laptop prices based on RAM and storage.
   - **Random Forest Regression:** Another model for price prediction, with hyperparameter tuning using `GridSearchCV` to find the best model configuration.

5. **Model Evaluation:**
   - Computes and compares RMSE (Root Mean Squared Error) and R^2 (Coefficient of Determination) for both models.
   - Outputs the best parameters found by `GridSearchCV` for the Random Forest model.

6. **Dependencies:**
   - `requests`, `BeautifulSoup` for web scraping.
   - `pandas` for data manipulation.
   - `matplotlib` and `seaborn` for data visualization.
   - `scikit-learn` for machine learning models and evaluation metrics.

#### Usage:

- Clone the repository and execute the Python script.
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- The script fetches data from Amazon.in, performs EDA, trains ML models, and outputs model performance metrics.

#### Files:

- `amazon_laptops_raw.csv`: Initial scraped data.
- `cleaned_amazon_laptops.csv`: Cleaned dataset after handling missing values.
- Python script (`laptop_price_prediction.py` or similar) containing the entire workflow.

This repository provides a comprehensive analysis pipeline from web scraping to machine learning model deployment, focusing on laptop price prediction based on key specifications.
