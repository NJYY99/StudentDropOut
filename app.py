import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.title("Student Dropout Analysis")
    
    # Display the dataset
    st.write(df)

    # Display dataset info
    st.write("Dataset Info:")
    st.write(df.info())

    ## Outlier Detection
    st.header("Outlier Detection")
    st.write("Pada bagian ini, kita akan mengidentifikasi outlier dalam dataset menggunakan metode Interquartile Range (IQR).")
    
    # Calculate Interquartile Range (IQR) for numerical columns
    q1 = df.select_dtypes(exclude=['object', 'bool']).quantile(0.25)
    q3 = df.select_dtypes(exclude=['object', 'bool']).quantile(0.75)
    iqr = q3 - q1
    st.write("IQR:", iqr)

    # Calculate lower and upper bounds for outlier detection
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    st.write("Lower Bounds:", lower_bound)
    st.write("Upper Bounds:", upper_bound)

    # Identify outliers
    df_numeric = df.select_dtypes(exclude=['object', 'bool'])
    outlier_filter = (df_numeric < lower_bound) | (df_numeric > upper_bound)
    st.write("Outliers Filter:", outlier_filter)

    # Display outlier counts
    st.write("Outlier Counts:")
    for col in outlier_filter.columns:
        if df[col].dtype != object:
            value_counts = outlier_filter[col].value_counts()
            st.write(f"Column: {col}")
            for value, count in value_counts.items():
                st.write(f"  Value: {value}, Count: {count}")
            st.write('-' * 30)

    # Analyze columns with outliers
    columns_with_outliers = [
        'Age', 'Travel_Time', 'Study_Time', 'Number_of_Failures',
        'Family_Relationship', 'Free_Time', 'Weekend_Alcohol_Consumption',
        'Number_of_Absences', 'Grade_1', 'Grade_2', 'Final_Grade'
    ]

    # Calculate outlier percentages
    outlier_percentages = {column: len(outlier_filter[outlier_filter[column]]) / len(df[column]) * 100 for column in columns_with_outliers}
    st.write("Outlier Percentages:", outlier_percentages)

    ## Visualization of Outliers
    st.header("Outlier Visualization")
    st.write("Visualisasi Boxplots berikut memvisualisasikan distribusi setiap fitur numerik dan menyorot setiap outlier potensial")
    for column in df_numeric:
        plt.figure(figsize=(20, 2))
        sns.boxplot(data=df_numeric, x=column)
        st.pyplot(plt)

    ## Family Support Analysis
    st.header("Family Support Analysis")
    st.write("Pada Bagian ini, saya menganalisis hubungan antara dukungan keluarga dan status kesehatan di kalangan siswa.")
    
    family_support_count = df.groupby('Family_Support').sum()['Health_Status'].sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    colors = sns.color_palette('pastel')[0:5]
    family_support_count.plot(kind='bar', color=colors)
    plt.ylabel('Jumlah Yang Sehat')
    plt.title('Jumlah Kesehatan berdasarkan dukungan keluarga')
    st.pyplot(plt)

    ## Correlation Analysis
    st.header("Correlation Analysis")
    st.write("Dilanjutkan dengan memeriksa korelasi antara fitur yang dipilih untuk memahami hubungannya.")
    
    factors = ['Study_Time', 'Family_Support', 'Extra_Curricular_Activities', 'Final_Grade']
    df_factors = df[factors]
    df_factors['Family_Support'] = df_factors['Family_Support'].map({'yes': 1, 'no': 0})
    df_factors['Extra_Curricular_Activities'] = df_factors['Extra_Curricular_Activities'].map({'yes': 1, 'no': 0})

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_factors.corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasi Faktor yang mempengaruhi nilai akhir')
    st.pyplot(plt)

    ## Family Support Distribution
    st.header("Family Support Distribution")
    st.write("Pie Chart Berikut menggambarkan distribusi dukungan keluarga di antara para siswa.")
    
    family_support_counts = df['Family_Support'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(family_support_counts, labels=family_support_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Family Support Distribution')
    plt.axis('equal')
    st.pyplot(plt)

    ## Handling Missing Data
    st.header("Missing Data Analysis")
    st.write("Mengecek Missing Value yang ada pada dataset.")
    st.write(df.isna().sum())

    ## Outlier Handling by Trimming
    st.header("Outlier Handling")
    st.write("MengTrim dataset untuk menghilangkan outlier yang sudah teridentifikasi sebelumnya.")
    
    df_cleaned = df[~outlier_filter.any(axis=1)]
    st.write("Cleaned Data Size:", df_cleaned.shape)

    ## Boxplots for Cleaned Data
    st.header("Boxplots for Cleaned Data")
    st.write("Boxplot berikut merupakan update dari hasil Handling Outlier.")
    for column in df_cleaned.columns:
        plt.figure(figsize=(20, 2))
        sns.boxplot(data=df_cleaned, x=column)
        plt.title(f'Boxplot for {column}')
        st.pyplot(plt)

    ## Scaling
    st.header("Scaling")
    st.write("Pada bagian ini, saya menstandardisasi dan menormalkan fitur numerik yang dipilih untuk meningkatkan kinerja model.")
    
    columns_to_scale = ['Travel_Time', 'Study_Time', 'Free_Time', 'Grade_1', 'Grade_2', 'Final_Grade']

    # Standard Scaling
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    st.write("Standardized Data:")
    st.write(df_scaled.head())

    ## Normalization
    max_min_scaled = (df[columns_to_scale] - df[columns_to_scale].min(axis=0)) / (df[columns_to_scale].max(axis=0) - df[columns_to_scale].min(axis=0))
    df[columns_to_scale] = max_min_scaled
    st.write("Normalized Data:")
    st.write(df.head())

    ## Label Encoding
    st.header("Label Encoding")
    st.write("Fitur/Tipe Data Kategorikal akan diubah menjadi format numerik menggunakan pengkodean label.")
    
    categorical_columns = [
        'School', 'Gender', 'Address', 'Family_Size', 'Parental_Status',
        'Mother_Job', 'Father_Job', 'Reason_for_Choosing_School',
        'Guardian', 'School_Support', 'Family_Support',
        'Extra_Paid_Class', 'Extra_Curricular_Activities',
        'Attended_Nursery', 'Wants_Higher_Education',
        'Internet_Access', 'In_Relationship'
    ]
    le = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(le.fit_transform)
    st.write("Label Encoded Data:")
    st.write(df.head())

    ## Logistic Regression Model
    st.header("Logistic Regression")
    st.write("Membangun model regresi logistik untuk memprediksi putus sekolah siswa berdasarkan fitur yang disiapkan.")
    
    df_model = df.copy()
    df_model = df_model.drop(df_model.columns[5], axis=1)  # Adjust this according to your dataset

    X = df_model.iloc[:, :-1].values
    y = df_model.iloc[:, -1].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Train the Logistic Regression model
    reg = LogisticRegression(solver='lbfgs', max_iter=1000)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    ## Confusion Matrix
    st.write("Confusion Matrix:")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    st.write(cnf_matrix)

    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

    ## Classification Report
    st.header("Classification Report")
    st.write("Rincian Laporan Classifikasi")
    
    target_names = ['Not Delayed', 'Delayed']
    report = metrics.classification_report(y_test, y_pred, target_names=target_names)
    st.text(report)

    ## ROC Curve
    st.write("Kurva ROC Berikut menggambarkan pertentangan antara sensitivitas dan spesifisitas untuk model.")
    
    y_pred_proba = reg.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="Score AUC=" + str(auc))
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    st.pyplot(plt)
