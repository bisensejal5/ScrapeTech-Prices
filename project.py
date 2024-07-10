import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract laptop data from a single page on Amazon
def fetch_amazon_data(page):
    url = f"https://www.amazon.in/s?k=laptop&page={page}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the data for each laptop
    laptops = []
    for item in soup.find_all("div", class_="s-result-item"):
        # Title
        title = item.find("span", class_="a-size-medium")
        if not title:
            continue
        title = title.text

        # Price
        price = item.find("span", class_="a-price-whole")
        if not price:
            continue
        price = float(price.text.replace(",", "").replace("₹", "").strip())  # Convert to float

        # Rating
        rating = item.find("span", class_="a-icon-alt")
        if rating:
            rating = float(rating.text.split()[0])
        else:
            rating = None

        # Specifications
        specs = item.find("div", class_="a-section a-spacing-none a-spacing-top-small")
        if specs:
            spec_dict = {}
            for spec in specs.find_all("span", class_="a-size-base"):
                if ":" in spec.text:
                    key, value = spec.text.split(":", 1)
                    spec_dict[key.strip()] = value.strip()
        else:
            spec_dict = {}

        laptop = {"title": title, "price": price, "rating": rating}
        laptop.update(spec_dict)
        laptops.append(laptop)

    return laptops

# Function to fetch data from multiple pages
def fetch_all_data(pages):
    all_laptops = []
    for page in range(1, pages + 1):
        laptops = fetch_amazon_data(page)
        all_laptops.extend(laptops)
        print(f"Page {page} scraped, {len(laptops)} laptops found.")
    return all_laptops

# Fetch data from the first 20 pages
laptops_data = fetch_all_data(20)

# Convert to DataFrame
laptops_df = pd.DataFrame(laptops_data)

# Save the scraped data to a CSV file
laptops_df.to_csv('amazon_laptops_raw.csv', index=False)

# Load the scraped data
laptops_df = pd.read_csv('amazon_laptops_raw.csv')

# Handle missing values and inconsistent data
laptops_df.dropna(subset=['price'], inplace=True)

# Function to extract RAM and Storage specifications
def extract_ram_storage(spec_text):
    ram = None
    storage = None
    if 'RAM' in spec_text:
        ram = spec_text.split('RAM')[0].strip()
    if 'SSD' in spec_text or 'HDD' in spec_text:
        storage = spec_text.split('SSD')[-1].strip()
    return ram, storage

# Apply extraction function to each row
laptops_df['RAM'], laptops_df['Storage'] = zip(*laptops_df['title'].apply(extract_ram_storage))

# Drop rows with missing RAM or Storage information
laptops_df.dropna(subset=['RAM', 'Storage'], inplace=True)

# Convert RAM and Storage to numeric values
laptops_df['RAM'] = laptops_df['RAM'].str.extract('(\d+)').astype(float)
laptops_df['Storage'] = laptops_df['Storage'].str.extract('(\d+)').astype(float)

# Save the cleaned data
laptops_df.to_csv('cleaned_amazon_laptops.csv', index=False)

# EDA
# Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(laptops_df['price'], kde=True)
plt.title('Distribution of Laptop Prices')
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')
plt.show()

# Relationship between Price and RAM
plt.figure(figsize=(10, 6))
sns.scatterplot(x=laptops_df['RAM'], y=laptops_df['price'])
plt.title('Price vs RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (INR)')
plt.show()

# Relationship between Price and Storage
plt.figure(figsize=(10, 6))
sns.scatterplot(x=laptops_df['Storage'], y=laptops_df['price'])
plt.title('Price vs Storage')
plt.xlabel('Storage (GB)')
plt.ylabel('Price (INR)')
plt.show()

# Additional: Price vs RAM and Price vs Storage in one figure
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=laptops_df['RAM'], y=laptops_df['price'])
plt.title('Price vs RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (INR)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=laptops_df['Storage'], y=laptops_df['price'])
plt.title('Price vs Storage')
plt.xlabel('Storage (GB)')
plt.ylabel('Price (INR)')

plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
corr_matrix = laptops_df[['price', 'RAM', 'Storage']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Features and Target
X = laptops_df[['RAM', 'Storage']]
y = laptops_df['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("Linear Regression R^2:", r2_score(y_test, y_pred_lr))

# Train a Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("Random Forest R^2:", r2_score(y_test, y_pred_rf))

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
print("Best Random Forest RMSE:", mean_squared_error(y_test, y_pred_best_rf, squared=False))
print("Best Random Forest R^2:", r2_score(y_test, y_pred_best_rf))
print("Best Parameters:", grid_search.best_params_)