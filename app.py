import io
import base64
from flask import Flask, render_template, request, send_file, url_for
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import matplotx
import glob
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether, Image

app = Flask(__name__)
app = Flask(__name__, static_folder='static')

app.jinja_env.globals.update(enumerate=enumerate)

def handle_missing_values(df, method='mean'):
    """Handles missing values using mean, median, or mode imputation."""
    for col in df.columns:
        if df[col].isnull().any():
            if method == 'mean':
                imputation_value = df[col].mean()
            elif method == 'median':
                imputation_value = df[col].median()
            elif method == 'mode':
                imputation_value = df[col].mode()[0]
            else:
                raise ValueError(
                    "Invalid imputation method. Use 'mean', 'median', or 'mode'."
                )
            df[col].fillna(imputation_value, inplace=True)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    csv_files = glob.glob('customer_data_*.csv')
    city_names = [os.path.basename(f).replace('customer_data_', '').replace('.csv', '') for f in csv_files]
    
    if request.method == 'POST':
        city_choice = int(request.form['city'])
        age_choice = int(request.form['age'])
        gender_choice = int(request.form['gender'])
        
        try:
            # Run your analysis
            data, graphs, conclusions = run_analysis(city_choice, age_choice, gender_choice)
            
            # Generate PDF
            generate_pdf(data, graphs, conclusions)
            
            # Instead of returning the PDF file directly, render the output template
            return render_template('output.html')
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}", 500
    
    return render_template('index.html', cities=city_names)

# Add a new route to download the PDF
@app.route('/download_pdf')
def download_pdf():
    return send_file('customer_analysis_report.pdf', as_attachment=True)

def run_analysis(city_choice, age_choice, gender_choice):
    print("Starting run_analysis function")
    graphs = []
    conclusions = []
    data = None
    
    csv_files = glob.glob('customer_data_*.csv')
    city_names = [os.path.basename(f).replace('customer_data_', '').replace('.csv', '') for f in csv_files]

    print(f"City choice: {city_choice}, Age choice: {age_choice}, Gender choice: {gender_choice}")
    print(f"Number of CSV files found: {len(csv_files)}")

    if city_choice <= len(city_names):
        data = pd.read_csv(csv_files[city_choice - 1])
    elif city_choice == len(city_names) + 1:
        data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        print("Invalid choice. Exiting...")
        return None, [], []

    print(f"Data shape after loading: {data.shape}")

    data = handle_missing_values(data, method='median')
    data_original = data.copy()
    print(f"Original data records: {len(data)}")

    # Age Group Filter
    if age_choice == 1:
        age_condition = (data['age'] >= 18) & (data['age'] <= 25)
    elif age_choice == 2:
        age_condition = (data['age'] >= 26) & (data['age'] <= 35)
    elif age_choice == 3:
        age_condition = (data['age'] >= 36) & (data['age'] <= 45)
    elif age_choice == 4:
        age_condition = (data['age'] >= 46) & (data['age'] <= 55)
    elif age_choice == 5:
        age_condition = (data['age'] >= 56) & (data['age'] <= 65)
    else:
        age_condition = True

    # Gender Based Filtering
    if gender_choice == 1:
        gender_condition = (data['gender'] == 'Male')
    elif gender_choice == 2:
        gender_condition = (data['gender'] == 'Female')
    else:
        gender_condition = True

    # Combine conditions and filter data
    if age_condition is not True and gender_condition is not True:
        data = data[age_condition & gender_condition]
    elif age_condition is not True:
        data = data[age_condition]
    elif gender_condition is not True:
        data = data[gender_condition]

    print(f"Data records after filtering: {len(data)}")

    # Checking if DataFrame is empty
    if data.empty:
        print("DataFrame is empty after filtering. Please adjust your filter criteria.")
        return None, [], []

    # Data preprocessing
    data['acquisitiondate'] = pd.to_datetime(data['acquisitiondate'])
    data['churn'] = data['churn'].astype(int)
    data['acquisitionmonth'] = data['acquisitiondate'].dt.month

    data['acquisitioncost'] = data['acquisitioncost'].str.replace('₹', '').str.replace(',', '').astype(float)
    data['totalspend'] = data['totalspend'].str.replace('₹', '').str.replace(',', '').astype(float)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(data[['marketingchannel', 'paymentmethod', 'billingissues']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['marketingchannel', 'paymentmethod', 'billingissues']))
    data = pd.concat([data, encoded_df], axis=1)

    # Outlier Removal (using IQR)
    numerical_features = ['retentionperiod', 'acquisitioncost', 'totalspend', 'customerinteractions', 'frequencyofbillingissues']
    for feature in numerical_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

    # Model training (if needed)
    features = ['acquisitionmonth', 'retentionperiod', 'age'] + list(encoded_df.columns)
    target = 'churn'

    X = data[features]
    y = data[target]

    if len(X) < 2 or len(np.unique(y)) < 2:
        print("Not enough data for training after filtering. Please adjust your criteria.")
        return None, [], []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Graph Generation
    with plt.style.context(matplotx.styles.pitaya_smoothie['dark']):
        # Customer Age Distribution (Histogram)
        plt.figure(figsize=(10, 5))
        plt.hist(data['age'], bins=10, edgecolor='black', color='#F25287', alpha=0.6)
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.grid(which='major', axis='y')
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()

        # Churn Rate by Age Group
        plt.figure(figsize=(10, 5))
        age_bins = [0, 25, 35, 45, 55, 65, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels,        right=False)
        churn_by_age = data.groupby('age_group')['churn'].mean()
        plt.bar(churn_by_age.index, churn_by_age.values, edgecolor='black', color='#F25287', alpha=0.6)
        plt.title('Churn Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Churn Rate')
        plt.grid(which='major', axis='y')
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()
       
        # Average Spend by Marketing Channel
        plt.figure(figsize=(10, 5))
        avg_spend = data.groupby('marketingchannel') ['totalspend'].mean().sort_values(ascending=False)
        plt.bar(avg_spend.index, avg_spend.values, edgecolor='black', color='#87F2CD', alpha=0.6)
        plt.title('Average Spend by Marketing Channel')
        plt.xlabel('Marketing Channel')
        plt.ylabel('Average Spend (₹)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(which='major', axis='y')
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()

        # Retention Period Distribution
        plt.figure(figsize=(10, 5))
        plt.hist(data['retentionperiod'], bins=20, edgecolor='black', color='#F25287', alpha=0.6)
        plt.title('Retention Period Distribution')
        plt.xlabel('Retention Period (days)')
        plt.ylabel('Frequency')
        plt.grid(which='major', axis='y')
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()

        # Customer Interactions vs. Churn
        plt.figure(figsize=(10, 5))
        interactions_churn = data.groupby('customerinteractions')['churn'].mean()
        plt.plot(interactions_churn.index, interactions_churn.values, marker='o', color='#F25287')
        plt.title('Customer Interactions vs. Churn Rate')
        plt.xlabel('Number of Customer Interactions')
        plt.ylabel('Churn Rate')
        plt.grid(True)
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()

        # Acquisition Cost vs. Total Spend
        plt.figure(figsize=(10, 5))
        plt.scatter(data['acquisitioncost'], data['totalspend'], alpha=0.5, color='#87F2CD')
        plt.title('Acquisition Cost vs. Total Spend')
        plt.xlabel('Acquisition Cost (₹)')
        plt.ylabel('Total Spend (₹)')
        plt.grid(True)
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()        

        # Customer Gender Distribution (Bar graph)
        plt.figure(figsize=(10, 5))
        gender_counts = data['gender'].value_counts()
        plt.bar(gender_counts.index, gender_counts.values, edgecolor='black', color=['#87F2CD', '#F25287'], alpha=0.6)
        plt.title('Customer Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Frequency')
        plt.grid(which='major', axis='y')
        for i, v in enumerate(gender_counts.values):
            plt.text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold')
        ax = plt.gca()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        graphs.append(plt.gcf())
        plt.close()

        
        # Add conclusions for each new graph
        conclusions.append("Conclusion (Age Disstribution): [Age Distribution]")
        conclusions.append("Conclusion (Churn Rate by Age Group): [Churn rate by age group]")
        conclusions.append("Conclusion (Average Spend by Marketing Channel): [Average Spend by Marketing Channel]")
        conclusions.append("Conclusion (Retention Period Distribution): [Retention Period Distribution]")
        conclusions.append("Conclusion (Customer Interactions vs. Churn): [Customer Interactions vs Churn]")
        conclusions.append("Conclusion (Acquisition Cost vs. Total Spend): [Acquisition Cost vs. Total Spend]")
        # Conclusion for gender distribution
        if 'Male' in gender_counts and 'Female' in gender_counts:
            if gender_counts['Male'] > gender_counts['Female']:
                conclusion = "There are more male customers than female customers."
            elif gender_counts['Male'] < gender_counts['Female']:
                conclusion = "There are more female customers than male customers."
            else:
                conclusion = "The number of male and female customers is roughly equal."
        else:
            conclusion = "Unable to determine gender distribution due to insufficient data."
        conclusions.append(f"Conclusion (Gender Distribution): {conclusion}")
        # Add more graph generation code here...

    print(f"Number of graphs generated: {len(graphs)}")
    print(f"Number of conclusions: {len(conclusions)}")

    return data, graphs, conclusions

def generate_pdf(data, graphs, conclusions):
    print("Starting generate_pdf function")
    print(f"Data shape: {data.shape}")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of conclusions: {len(conclusions)}")

    try:
        doc = SimpleDocTemplate("customer_analysis_report.pdf", pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = []

        # Select desired columns for the PDF table
        selected_columns = [
            'customerid', 'acquisitiondate', 'age', 'gender', 'churn',
            'retentionperiod', 'acquisitioncost', 'totalspend', 'marketingchannel',
            'customerinteractions', 'paymentmethod', 'billingissues',
            'frequencyofbillingissues'
        ]
        filtered_data = data[selected_columns]

        # Convert data to list of lists for ReportLab table
        data_list = [filtered_data.columns.tolist()]  # Add header row
        for row in filtered_data.values.tolist():
            row[0] = row[0][:8]  # Truncate customer ID to first 8 characters
            data_list.append(row)

        # Calculate the number of rows per page
        rows_per_page = 20

        # Create ReportLab table
        for i in range(0, len(data_list), rows_per_page):
            # Get data for the current page
            page_data = data_list[i:i + rows_per_page]

            # Add header row on each page
            if i > 0:
                page_data.insert(0, data_list[0])

            table = Table(page_data, colWidths=[100] + [50] * (len(page_data[0]) - 1), rowHeights=25, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#121212'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#F25287'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 6),
                ('FONTSIZE', (1, 0), (-1, -1), 4),
                ('FONTSIZE', (1, 0), (1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, 'black'),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))

            # Add page break after each page except the last one
            if i + rows_per_page < len(data_list):
                story.append(PageBreak())

        # New page for graphs and conclusions
        story.append(PageBreak())
        story.append(Paragraph("Graphs and Conclusions", styles['h1']))
        story.append(Spacer(1, 12))

        # Graphs and conclusions
        for i, graph in enumerate(graphs):
            # Use BytesIO to store the image data in memory
            img_data = io.BytesIO()
            graph.savefig(img_data, format='png')
            img_data.seek(0)  # Reset the buffer position

            story.append(Paragraph(f"Graph {i+1}", styles['h2']))
            story.append(Image(img_data, width=400, height=200))
            story.append(Spacer(1, 12))

            # Add conclusions for each graph (except the first one)
            if i < len(conclusions):
                story.append(Paragraph(conclusions[i], styles['Normal']))
                story.append(Spacer(1, 12))

            story.append(PageBreak())

        doc.build(story)
        print("Finished generate_pdf function")
    except Exception as e:
        print(f"Error in generate_pdf: {str(e)}")
if __name__ == '__main__':
    app.run(debug=True)