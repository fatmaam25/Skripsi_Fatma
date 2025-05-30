import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import io
import base64
import os
import pickle
import datetime

# Set page configS
st.set_page_config(
    page_title="Prediksi Panen Cabai",
    page_icon="🌶️",
    layout="wide"
)

# Initialize session state variables
if 'data_malang' not in st.session_state:
    st.session_state.data_malang = {}
    
    # Data Malang 2018
    st.session_state.data_malang[2018] = pd.DataFrame({
        'X1(CURAH HUJAN)': [389.20, 383.30, 260.50, 107.10, 161.70, 42.50, 14.30, 24.50, 34.70, 24.90, 62.00, 314.50],
        'X2(SUHU)': [25.40, 25.10, 26.10, 26.10, 25.20, 24.80, 24.40, 14.30, 24.50, 34.70, 24.90, 26.30],
        'X3(LUAS PANEN)': [1345, 1337, 1440, 1443, 1836, 1729, 1475, 1224, 1235, 1226, 626, 418],  
        'Y': [54680, 61003, 66760, 56227, 64898, 72915, 61818, 50128, 51703, 71415, 24970, 19797]  
    })
    
    # Data Malang 2019
    st.session_state.data_malang[2019] = pd.DataFrame({
        'X1(CURAH HUJAN)': [497.70, 263.30, 456.40, 194.00, 45.90, 27.90, 10.70, 24.40, 15.40, 24.30, 28.10, 70.70],
        'X2(SUHU)': [26.30, 26.50, 25.90, 26.10, 25.20, 24.20, 24.00, 10.70, 24.40, 15.40, 24.30, 21.40],
        'X3(LUAS PANEN)': [735, 1341, 1631, 1696, 1718, 1720, 1737, 1271, 641, 1205, 969, 845],  
        'Y': [17225, 44331, 59105, 74422, 68398, 80848, 94669, 71294, 43928, 74590, 33657, 31433]  
    })
    
    # Data Malang 2020
    st.session_state.data_malang[2020] = pd.DataFrame({
        'X1(CURAH HUJAN)': [333.20, 403.00, 316.30, 225.30, 224.30, 65.90, 27.10, 25.30, 19.90, 25.60, 118.30, 211.90],
        'X2(SUHU)': [26.70, 26.30, 26.10, 26.10, 25.20, 25.40, 25.20, 27.10, 25.30, 19.90, 25.60, 62.30],
        'X3(LUAS PANEN)': [813, 891, 1410, 1518, 1685, 694, 1639, 1337, 869, 1011, 224, 275],  
        'Y': [55438, 34461, 73339, 109417, 127267, 47328, 90489, 66846, 66953, 78647, 14665, 14473]  
    })
    
    # Data Malang 2021
    st.session_state.data_malang[2021] = pd.DataFrame({
        'X1(CURAH HUJAN)': [610.20, 301.20, 302.80, 160.50, 74.60, 127.40, 15.20, 25.30, 28.60, 25.70, 198.80, 430.00],
        'X2(SUHU)': [24.90, 25.30, 25.40, 25.70, 25.20, 25.30, 25.00, 15.20, 25.30, 28.60, 25.70, 169.30],
        'X3(LUAS PANEN)': [589, 496, 1119, 857, 682, 837, 1609, 732, 1421, 754, 723, 817],  
        'Y': [46804, 40857, 91761, 70551, 55821, 68567, 123734, 56399, 116502, 59524, 58080, 48025]  
    })
    
    # Data Malang 2022
    st.session_state.data_malang[2022] = pd.DataFrame({
        'X1(CURAH HUJAN)': [392.30, 333.30, 333.10, 186.70, 204.80, 177.80, 40.80, 25.40, 24.10, 25.60, 280.40, 485.70],
        'X2(SUHU)': [26.00, 24.60, 26.00, 26.10, 26.20, 25.40, 25.30, 40.80, 25.40, 24.10, 25.60, 68.10],
        'X3(LUAS PANEN)': [785, 1156, 768, 1800, 1782.50, 1617.50, 1263, 1342.50, 1003.40, 964.75, 1243.65, 1046.10],  
        'Y': [140155, 77812, 48133, 93932, 92549, 93874, 71467, 76721, 48753.80, 45802.50, 64340, 20798]  
    })

if 'data_lumajang' not in st.session_state:
    st.session_state.data_lumajang = {}
    
    # Data Lumajang 2018
    st.session_state.data_lumajang[2018] = pd.DataFrame({
        'X1(CURAH HUJAN)': [351.80, 316.80, 221.90, 101.00, 167.30, 52.40, 25.40, 80.00, 40.00, 93.20, 389.00, 252.70],
        'X2(SUHU)': [26.50, 26.00, 27.20, 27.00, 26.30, 26.00, 25.60, 25.90, 26.30, 27.40, 27.20, 27.80],
        'X3(LUAS PANEN)': [69, 37, 49, 43, 16, 25, 15, 21, 51, 55, 104, 826],  
        'Y': [3814.40, 6878.00, 6758.00, 4890.00, 3800.00, 11650.00, 21790.00, 28400.00, 27836.00, 36910.00, 31050.00, 21365.00]  
    })
    
    # Data Lumajang 2019
    st.session_state.data_lumajang[2019] = pd.DataFrame({
        'X1(CURAH HUJAN)': [381.30, 176.20, 442.90, 136.90, 45.40, 36.70, 18.20, 33.50, 38.60, 49.30, 79.40, 214.30],
        'X2(SUHU)': [27.40, 27.40, 26.90, 27.10, 26.30, 25.30, 25.20, 25.80, 25.70, 27.80, 27.50, 27.50],
        'X3(LUAS PANEN)': [9, 83, 74, 11, 19, 15, 7, 13, 14, 240, 108, 13],  
        'Y': [3616.00, 8279.00, 6967.80, 5174.00, 3653.00, 19730.50, 19048.43, 13467.80, 15129.00, 28488.00, 25209.00, 13714.00]  
    })
    
    # Data Lumajang 2020
    st.session_state.data_lumajang[2020] = pd.DataFrame({
        'X1(CURAH HUJAN)': [253.80, 294.30, 326.30, 197.50, 236.30, 84.30, 35.10, 27.50, 85.20, 191.70, 251.30, 406.60],
        'X2(SUHU)': [27.90, 27.30, 27.10, 27.10, 26.30, 26.60, 26.50, 26.70, 27.20, 27.60, 27.70, 27.10],
        'X3(LUAS PANEN)': [351, 278, 25, 13, 21, 12, 13, 9, 20, 236, 17, 1405],  
        'Y': [16008, 8880, 3105, 2690, 2456, 5085, 9470, 29231, 32002, 37643, 40349, 24650]  
    })
    
    # Data Lumajang 2021
    st.session_state.data_lumajang[2021] = pd.DataFrame({
        'X1(CURAH HUJAN)': [540.40, 267.00, 323.80, 146.20, 70.80, 175.40, 23.40, 57.80, 245.90, 317.60, 531.30, 593.70],
        'X2(SUHU)': [26.00, 26.20, 26.50, 26.70, 26.30, 26.50, 26.40, 26.80, 27.30, 27.80, 26.60, 27.20],
        'X3(LUAS PANEN)': [573, 3, 9, 27, 15, 8, 31, 20, 229, 24, 647, 437],  
        'Y': [4649, 1870, 2028, 3070, 2479, 2294, 3956, 23625, 21829, 8921, 20837.80, 30741.70]  
    })
    
    # Data Lumajang 2022
    st.session_state.data_lumajang[2022] = pd.DataFrame({
        'X1(CURAH HUJAN)': [331.40, 274.30, 323.00, 186.60, 209.30, 216.10, 51.50, 48.30, 109.90, 400.20, 598.40, 360.50],
        'X2(SUHU)': [27.10, 25.60, 27.10, 27.10, 27.40, 26.70, 26.60, 26.90, 27.20, 26.80, 26.50, 27.20],
        'X3(LUAS PANEN)': [20, 23, 26, 44, 23, 8, 7, 29, 251, 8, 15, 288],  
        'Y': [3336, 3537, 5115, 4952.10, 4167, 3992, 16831, 17072, 15251.75, 9992, 7959, 8140.80]  
    })

if 'models_malang' not in st.session_state:
    st.session_state.models_malang = {}

if 'models_lumajang' not in st.session_state:
    st.session_state.models_lumajang = {}

if 'rmse_malang' not in st.session_state:
    st.session_state.rmse_malang = {}

if 'rmse_lumajang' not in st.session_state:
    st.session_state.rmse_lumajang = {}

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize data structures for each year
for year in range(2018, 2023):
    if year not in st.session_state.data_malang:
        st.session_state.data_malang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])
    
    if year not in st.session_state.data_lumajang:
        st.session_state.data_lumajang[year] = pd.DataFrame(columns=['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)', 'Y'])

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# Function to save models
def save_model(city, year, model):
    if city == "Malang":
        st.session_state.models_malang[year] = model
    else:
        st.session_state.models_lumajang[year] = model
    
    # Calculate RMSE for next year if data exists
    next_year = year + 1
    if next_year <= 2022:
        rmse = evaluate_model(city, year, next_year)
        if rmse is not None:
            if city == "Malang":
                st.session_state.rmse_malang[next_year] = rmse
            else:
                st.session_state.rmse_lumajang[next_year] = rmse


# Function to train models
def train_model(city, year):
    if city == "Malang":
        data = st.session_state.data_malang[year]
    else:
        data = st.session_state.data_lumajang[year]
    
    if len(data) < 3:  # Need at least 3 data points for meaningful regression
        return None
    
    X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
    y = data['Y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model


def evaluate_model(city, model_year, eval_year):
    """
    Evaluates a model by using model_year's model to predict eval_year's data
    and calculates RMSE
    """
    if city == "Malang":
        if model_year not in st.session_state.models_malang:
            return None
        if eval_year not in st.session_state.data_malang:
            return None
        model = st.session_state.models_malang[model_year]
        eval_data = st.session_state.data_malang[eval_year]
    else:
        if model_year not in st.session_state.models_lumajang:
            return None
        if eval_year not in st.session_state.data_lumajang:
            return None
        model = st.session_state.models_lumajang[model_year]
        eval_data = st.session_state.data_lumajang[eval_year]
    
    X_eval = eval_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
    y_actual = eval_data['Y']
    
    y_pred = model.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    return rmse


# Function to make predictions
def predict(city, model_year, curah_hujan, suhu, luas_panen):
    if city == "Malang":
        if model_year not in st.session_state.models_malang:
            return None
        model = st.session_state.models_malang[model_year]
    else:
        if model_year not in st.session_state.models_lumajang:
            return None
        model = st.session_state.models_lumajang[model_year]
    
    prediction = model.predict([[curah_hujan, suhu, luas_panen]])
    return prediction[0]

# Function to download data as CSV
def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Main app header
st.markdown(
    """
    <h1 style='text-align: center;'>🌶️ Sistem Prediksi Panen Cabai</h1>
    <p style='text-align: center;'>Prediksi hasil panen cabai menggunakan regresi linear berganda untuk Kabupaten Malang dan Lumajang</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Menu Navigasi")
dashboard_btn = st.sidebar.button("Dashboard", use_container_width=True)
data_btn = st.sidebar.button("Data Aktual", use_container_width=True)
predict_btn = st.sidebar.button("Prediksi", use_container_width=True)
results_btn = st.sidebar.button("Hasil", use_container_width=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

if dashboard_btn:
    st.session_state.current_page = "Dashboard"
if data_btn:
    st.session_state.current_page = "Data Aktual"
if predict_btn:
    st.session_state.current_page = "Prediksi"
if results_btn:
    st.session_state.current_page = "Hasil"

# Dashboard page
if st.session_state.current_page == "Dashboard":
    st.header("Dashboard Prediksi Panen Cabai")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kabupaten Malang")
        city_data = pd.DataFrame()
        
        for year in range(2018, 2023):
            if not st.session_state.data_malang[year].empty:
                temp_df = st.session_state.data_malang[year].copy()
                temp_df['Tahun'] = year
                temp_df['Bulan'] = range(1, len(temp_df) + 1)
                city_data = pd.concat([city_data, temp_df])
        
        if not city_data.empty:
            # Show yearly production trend
            st.markdown("### Trend Produksi Tahunan")
            yearly_production = city_data.groupby('Tahun')['Y'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(yearly_production['Tahun'], yearly_production['Y'], color='green')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Total Produksi')
            ax.set_title('Total Produksi Cabai per Tahun - Malang')
            st.pyplot(fig)
            
        else:
            st.info("Belum ada data untuk Malang. Silakan tambahkan data di menu 'Data Aktual'.")
    
    with col2:
        st.subheader("Kota Lumajang")
        city_data = pd.DataFrame()
        
        for year in range(2018, 2023):
            if not st.session_state.data_lumajang[year].empty:
                temp_df = st.session_state.data_lumajang[year].copy()
                temp_df['Tahun'] = year
                temp_df['Bulan'] = range(1, len(temp_df) + 1)
                city_data = pd.concat([city_data, temp_df])
        
        if not city_data.empty:
            # Show yearly production trend
            st.markdown("### Trend Produksi Tahunan")
            yearly_production = city_data.groupby('Tahun')['Y'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(yearly_production['Tahun'], yearly_production['Y'], color='blue')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Total Produksi')
            ax.set_title('Total Produksi Cabai per Tahun - Lumajang')
            st.pyplot(fig)
            
            # Show RMSE comparison if models exist
            if st.session_state.rmse_lumajang:
                st.markdown("### Akurasi Model (RMSE)")
                rmse_data = pd.DataFrame({
                    'Tahun': list(st.session_state.rmse_lumajang.keys()),
                    'RMSE': list(st.session_state.rmse_lumajang.values())
                })
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(rmse_data['Tahun'], rmse_data['RMSE'], color='orange')
                ax.set_xlabel('Tahun')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE per Tahun - Lumajang')
                st.pyplot(fig)
        else:
            st.info("Belum ada data untuk Lumajang. Silakan tambahkan data di menu 'Data Aktual'.")
    
    # Show comparison between cities
    st.header("Perbandingan Produksi Antar Kota")
    
    malang_data = pd.DataFrame()
    lumajang_data = pd.DataFrame()
    
    for year in range(2018, 2023):
        if not st.session_state.data_malang[year].empty:
            temp_df = st.session_state.data_malang[year].copy()
            temp_df['Tahun'] = year
            malang_data = pd.concat([malang_data, temp_df])
        
        if not st.session_state.data_lumajang[year].empty:
            temp_df = st.session_state.data_lumajang[year].copy()
            temp_df['Tahun'] = year
            lumajang_data = pd.concat([lumajang_data, temp_df])
    
    if not malang_data.empty and not lumajang_data.empty:
        malang_yearly = malang_data.groupby('Tahun')['Y'].sum().reset_index()
        malang_yearly['Kota'] = 'Malang'
        
        lumajang_yearly = lumajang_data.groupby('Tahun')['Y'].sum().reset_index()
        lumajang_yearly['Kota'] = 'Lumajang'
        
        combined_data = pd.concat([malang_yearly, lumajang_yearly])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Tahun', y='Y', hue='Kota', data=combined_data, ax=ax)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Total Produksi')
        ax.set_title('Perbandingan Produksi Cabai Antar Kota')
        st.pyplot(fig)
    else:
        st.info("Data tidak cukup untuk perbandingan antar kota.")

# Data Aktual page
elif st.session_state.current_page == "Data Aktual":
    st.header("Data Aktual Panen Cabai")
    
    tab1, tab2 = st.tabs(["Malang", "Lumajang"])
    
    with tab1:
        st.subheader("Data Kabupaten Malang")
        year_malang = st.selectbox("Pilih Tahun (Malang)", list(range(2018, 2023)), key='year_malang')
        
        # CRUD operations
        st.markdown("### Manage Data")
        crud_tabs = st.tabs(["Lihat", "Tambah", "Edit", "Hapus"])
        
        with crud_tabs[0]:  # View
            if not st.session_state.data_malang[year_malang].empty:
                st.dataframe(st.session_state.data_malang[year_malang])
                st.markdown(get_download_link(st.session_state.data_malang[year_malang], f"malang_{year_malang}.csv"), unsafe_allow_html=True)
            else:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
        
        with crud_tabs[1]:  # Add
            st.markdown("### Tambah Data Baru")
            with st.form("add_data_malang"):
                month = st.selectbox("Bulan", list(range(1, 13)))
                curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f")
                suhu = st.number_input("Suhu", min_value=0.0, format="%.2f")
                luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f")
                hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, format="%.3f")
                
                submit_button = st.form_submit_button("Tambah Data")
                if submit_button:
                    # Check if month already exists
                    if len(st.session_state.data_malang[year_malang]) >= month:
                        st.error(f"Data untuk bulan {month} sudah ada. Gunakan Edit untuk mengubah data.")
                    else:
                        new_data = pd.DataFrame({
                            'X1(CURAH HUJAN)': [curah_hujan],
                            'X2(SUHU)': [suhu],
                            'X3(LUAS PANEN)': [luas_panen],
                            'Y': [hasil_panen]
                        })
                        st.session_state.data_malang[year_malang] = pd.concat([st.session_state.data_malang[year_malang], new_data], ignore_index=True)
                        st.success("Data berhasil ditambahkan!")

        with crud_tabs[2]:  # Edit
            st.markdown("### Edit Data")
            if st.session_state.data_malang[year_malang].empty:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
            else:
                month_to_edit = st.selectbox("Pilih Bulan untuk Edit", list(range(1, len(st.session_state.data_malang[year_malang]) + 1)), key='edit_month_malang')
                
                idx = month_to_edit - 1
                if idx < len(st.session_state.data_malang[year_malang]):
                    with st.form("edit_data_malang"):
                        curah_hujan = st.number_input("Curah Hujan", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X1(CURAH HUJAN)']), format="%.2f")
                        suhu = st.number_input("Suhu", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X2(SUHU)']), format="%.2f")
                        luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['X3(LUAS PANEN)']), format="%.3f")
                        hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, value=float(st.session_state.data_malang[year_malang].iloc[idx]['Y']), format="%.3f")
                        
                        submit_button = st.form_submit_button("Update Data")
                        if submit_button:
                            st.session_state.data_malang[year_malang].iloc[idx] = [curah_hujan, suhu, luas_panen, hasil_panen]
                            st.success("Data berhasil diupdate!")
                            
                            # Retrain model after editing data
                            model = train_model("Malang", year_malang)
                            if model is not None:
                                save_model("Malang", year_malang, model)
                                next_year = year_malang + 1
                                if next_year in st.session_state.rmse_malang:
                                    st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                                else:
                                    st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang!")
        
        with crud_tabs[3]:  # Delete
            st.markdown("### Hapus Data")
            if st.session_state.data_malang[year_malang].empty:
                st.info(f"Belum ada data untuk tahun {year_malang}.")
            else:
                month_to_delete = st.selectbox("Pilih Bulan untuk Hapus", list(range(1, len(st.session_state.data_malang[year_malang]) + 1)), key='delete_month_malang')
                
                if st.button("Hapus Data"):
                    idx = month_to_delete - 1
                    if idx < len(st.session_state.data_malang[year_malang]):
                        st.session_state.data_malang[year_malang] = st.session_state.data_malang[year_malang].drop(st.session_state.data_malang[year_malang].index[idx]).reset_index(drop=True)
                        st.success("Data berhasil dihapus!")
                        
                        # Retrain model after deleting data
                        model = train_model("Malang", year_malang)
                        if model is not None:
                            save_model("Malang", year_malang, model)
                            next_year = year_malang + 1
                            if next_year in st.session_state.rmse_malang:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_malang} berhasil dilatih ulang!")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Malang"):
            success_count = 0
            for year in range(2018, 2022):  # Only train models for 2018-2021 (to predict 2019-2022)
                if not st.session_state.data_malang[year].empty and len(st.session_state.data_malang[year]) >= 3:
                    model = train_model("Malang", year)
                    if model is not None:
                        save_model("Malang", year, model)
                        success_count += 1
                        next_year = year + 1
                        if next_year in st.session_state.rmse_malang:
                            st.success(f"Model untuk tahun {year} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_malang[next_year]:.4f}")
                        else:
                            st.success(f"Model untuk tahun {year} berhasil dilatih!")
            
            if success_count > 0:
                st.success(f"{success_count} model berhasil dilatih untuk Malang!")
            else:
                st.warning("Tidak ada model yang berhasil dilatih. Pastikan data cukup (minimal 3 baris per tahun).")

    
    with tab2:
        st.subheader("Data Kota Lumajang")
        year_lumajang = st.selectbox("Pilih Tahun (Lumajang)", list(range(2018, 2023)), key='year_lumajang')
        
        # CRUD operations
        st.markdown("### Manage Data")
        crud_tabs = st.tabs(["Lihat", "Tambah", "Edit", "Hapus"])
        
        with crud_tabs[0]:  # View
            if not st.session_state.data_lumajang[year_lumajang].empty:
                st.dataframe(st.session_state.data_lumajang[year_lumajang])
                st.markdown(get_download_link(st.session_state.data_lumajang[year_lumajang], f"lumajang_{year_lumajang}.csv"), unsafe_allow_html=True)
            else:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
        
        with crud_tabs[1]:  # Add
            st.markdown("### Tambah Data Baru")
            with st.form("add_data_lumajang"):
                month = st.selectbox("Bulan", list(range(1, 13)), key='month_add_lumajang')
                curah_hujan = st.number_input("Curah Hujan", min_value=0.0, format="%.2f", key='ch_add_lumajang')
                suhu = st.number_input("Suhu", min_value=0.0, format="%.2f", key='suhu_add_lumajang')
                luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, format="%.3f", key='luas_add_lumajang')
                hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, format="%.3f", key='hasil_add_lumajang')
                
                submit_button = st.form_submit_button("Tambah Data")
                if submit_button:
                    # Check if month already exists
                    if len(st.session_state.data_lumajang[year_lumajang]) >= month:
                        st.error(f"Data untuk bulan {month} sudah ada. Gunakan Edit untuk mengubah data.")
                    else:
                        new_data = pd.DataFrame({
                            'X1(CURAH HUJAN)': [curah_hujan],
                            'X2(SUHU)': [suhu],
                            'X3(LUAS PANEN)': [luas_panen],
                            'Y': [hasil_panen]
                        })
                        st.session_state.data_lumajang[year_lumajang] = pd.concat([st.session_state.data_lumajang[year_lumajang], new_data], ignore_index=True)
                        st.success("Data berhasil ditambahkan!")
                        
                        # # Train model after adding data
                        # model = train_model("Lumajang", year_lumajang)
                        # if model is not None:
                        #     save_model("Lumajang", year_lumajang, model)
                        #     next_year = year_lumajang + 1
                        #     if next_year in st.session_state.rmse_lumajang:
                        #         st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                        #     else:
                        #         st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih!")
        
        with crud_tabs[2]:  # Edit
            st.markdown("### Edit Data")
            if st.session_state.data_lumajang[year_lumajang].empty:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
            else:
                month_to_edit = st.selectbox("Pilih Bulan untuk Edit", list(range(1, len(st.session_state.data_lumajang[year_lumajang]) + 1)), key='edit_month_lumajang')
                
                idx = month_to_edit - 1
                if idx < len(st.session_state.data_lumajang[year_lumajang]):
                    with st.form("edit_data_lumajang"):
                        curah_hujan = st.number_input("Curah Hujan", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X1(CURAH HUJAN)']), format="%.2f", key='ch_edit_lumajang')
                        suhu = st.number_input("Suhu", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X2(SUHU)']), format="%.2f", key='suhu_edit_lumajang')
                        luas_panen = st.number_input("Luas Panen (ha)", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['X3(LUAS PANEN)']), format="%.3f", key='luas_edit_lumajang')
                        hasil_panen = st.number_input("Hasil Panen (ton)", min_value=0.0, value=float(st.session_state.data_lumajang[year_lumajang].iloc[idx]['Y']), format="%.3f", key='hasil_edit_lumajang')
                        
                        submit_button = st.form_submit_button("Update Data")
                        if submit_button:
                            st.session_state.data_lumajang[year_lumajang].iloc[idx] = [curah_hujan, suhu, luas_panen, hasil_panen]
                            st.success("Data berhasil diupdate!")
                            
                            # Retrain model after editing data
                            model = train_model("Lumajang", year_lumajang)
                            if model is not None:
                                save_model("Lumajang", year_lumajang, model)
                                next_year = year_lumajang + 1
                                if next_year in st.session_state.rmse_lumajang:
                                    st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                                else:
                                    st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang!")
        
        with crud_tabs[3]:  # Delete
            st.markdown("### Hapus Data")
            if st.session_state.data_lumajang[year_lumajang].empty:
                st.info(f"Belum ada data untuk tahun {year_lumajang}.")
            else:
                month_to_delete = st.selectbox("Pilih Bulan untuk Hapus", list(range(1, len(st.session_state.data_lumajang[year_lumajang]) + 1)), key='delete_month_lumajang')
                
                if st.button("Hapus Data", key="delete_btn_lumajang"):
                    idx = month_to_delete - 1
                    if idx < len(st.session_state.data_lumajang[year_lumajang]):
                        st.session_state.data_lumajang[year_lumajang] = st.session_state.data_lumajang[year_lumajang].drop(st.session_state.data_lumajang[year_lumajang].index[idx]).reset_index(drop=True)
                        st.success("Data berhasil dihapus!")
                        
                        # Retrain model after deleting data
                        model = train_model("Lumajang", year_lumajang)
                        if model is not None:
                            save_model("Lumajang", year_lumajang, model)
                            next_year = year_lumajang + 1
                            if next_year in st.session_state.rmse_lumajang:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                            else:
                                st.success(f"Model untuk tahun {year_lumajang} berhasil dilatih ulang!")
        
        # Train all models button
        if st.button("Latih Semua Model untuk Lumajang"):
            success_count = 0
            for year in range(2018, 2022):  # Only train models for 2018-2021 (to predict 2019-2022)
                if not st.session_state.data_lumajang[year].empty and len(st.session_state.data_lumajang[year]) >= 3:
                    model = train_model("Lumajang", year)
                    if model is not None:
                        save_model("Lumajang", year, model)
                        success_count += 1
                        next_year = year + 1
                        if next_year in st.session_state.rmse_lumajang:
                            st.success(f"Model untuk tahun {year} berhasil dilatih! RMSE untuk prediksi tahun {next_year}: {st.session_state.rmse_lumajang[next_year]:.4f}")
                        else:
                            st.success(f"Model untuk tahun {year} berhasil dilatih!")
            
            if success_count > 0:
                st.success(f"{success_count} model berhasil dilatih untuk Lumajang!")
            else:
                st.warning("Tidak ada model yang berhasil dilatih. Pastikan data cukup (minimal 3 baris per tahun).")

# Prediksi page
elif st.session_state.current_page == "Prediksi":
    st.header("Prediksi Panen Cabai")
    
    # Step 1: Pilih Data Aktual untuk Prediksi
    st.subheader("Step 1: Pilih Data Aktual untuk Prediksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox("Pilih Kota", ["Malang", "Lumajang"])
    
    with col2:
        available_years = []
        if city == "Malang":
            for year in range(2018, 2023):
                if not st.session_state.data_malang[year].empty and len(st.session_state.data_malang[year]) >= 3:
                    available_years.append(year)
        else:
            for year in range(2018, 2023):
                if not st.session_state.data_lumajang[year].empty and len(st.session_state.data_lumajang[year]) >= 3:
                    available_years.append(year)
        
        if not available_years:
            st.error(f"Belum ada data yang cukup untuk kota {city}. Minimal 3 data per tahun diperlukan.")
            st.stop()
        
        selected_data_year = st.selectbox("Pilih Tahun Data untuk Model", available_years)
    
    # Step 2: Tentukan Tahun yang akan Diprediksi
    st.subheader("Step 2: Tentukan Tahun yang akan Diprediksi")
    
    prediction_year = st.selectbox("Pilih Tahun yang akan Diprediksi", 
                                 [year for year in range(2019, 2024) if year > selected_data_year])
    
    # Step 3: Klik Prediksi 
    st.subheader("Step 3: Hasil Tabel Penolong dan Persamaan prediksi")
    
    if st.button("Lakukan Prediksi"):
        if city == "Malang":
            data = st.session_state.data_malang[selected_data_year]
        else:
            data = st.session_state.data_lumajang[selected_data_year]
        
        # Siapkan data training
        X = data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
        y = data['Y']
        
        # Train model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Untuk prediksi, misal gunakan rata-rata dari data training sebagai input prediksi
        input_curah_hujan = X['X1(CURAH HUJAN)'].mean()
        input_suhu = X['X2(SUHU)'].mean()
        input_luas_panen = X['X3(LUAS PANEN)'].mean()
        
        prediction = model.predict([[input_curah_hujan, input_suhu, input_luas_panen]])[0]
        
        # Tampilkan Tabel Penolong
        df_helper = data.copy()
        df_helper["X12"] = df_helper["X1(CURAH HUJAN)"] ** 2
        df_helper["X22"] = df_helper["X2(SUHU)"] ** 2
        df_helper["X32"] = df_helper["X3(LUAS PANEN)"] ** 2
        df_helper["Y2"] = df_helper["Y"] ** 2
        df_helper["X1Y"] = df_helper["X1(CURAH HUJAN)"] * df_helper["Y"]
        df_helper["X2Y"] = df_helper["X2(SUHU)"] * df_helper["Y"]
        df_helper["X3Y"] = df_helper["X3(LUAS PANEN)"] * df_helper["Y"]
        df_helper["X1X2"] = df_helper["X1(CURAH HUJAN)"] * df_helper["X2(SUHU)"]
        df_helper["X1X3"] = df_helper["X1(CURAH HUJAN)"] * df_helper["X3(LUAS PANEN)"]
        df_helper["X2X3"] = df_helper["X2(SUHU)"] * df_helper["X3(LUAS PANEN)"]
        
        month_labels = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        if len(df_helper) <= len(month_labels):
            df_helper.index = month_labels[:len(df_helper)]
        else:
            df_helper.index = [f"Data {i+1}" for i in range(len(df_helper))]
        
        st.dataframe(df_helper)
        
        # Simpan hasil prediksi ke session_state
        prediction_key = f"{city}_{selected_data_year}_{prediction_year}"
        st.session_state.prediction_results[prediction_key] = {
            'city': city,
            'data_year': selected_data_year,
            'prediction_year': prediction_year,
            'model': model,
            'training_data': data,
            'input_data': {
                'curah_hujan': input_curah_hujan,
                'suhu': input_suhu,
                'luas_panen': input_luas_panen
            },
            'prediction': prediction,
            'timestamp': datetime.datetime.now()
        }
        
        # st.success(f"Prediksi berhasil! Hasil prediksi untuk tahun {prediction_year}: {prediction:.3f} ton")
        
        # Tampilkan persamaan regresi
        st.subheader("Persamaan Regresi")
        coefficients = model.coef_
        intercept = model.intercept_
        
        st.markdown(f"**Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃**")
        st.markdown(f"- a (konstanta) = {intercept:.4f}")
        st.markdown(f"- b₁ (koefisien curah hujan) = {coefficients[0]:.4f}")
        st.markdown(f"- b₂ (koefisien suhu) = {coefficients[1]:.4f}")
        st.markdown(f"- b₃ (koefisien luas panen) = {coefficients[2]:.4f}")

        st.info("Hasil lengkap prediksi dapat dilihat di menu 'Hasil'")


# Hasil page
elif st.session_state.current_page == "Hasil":
    st.header("Hasil Prediksi")
    
    if not st.session_state.prediction_results:
        st.info("Belum ada prediksi yang dilakukan. Silakan lakukan prediksi terlebih dahulu di menu 'Prediksi'.")
    else:
        prediction_options = [
            (key, f"{value['city']} - Model {value['data_year']} untuk prediksi {value['prediction_year']}")
            for key, value in st.session_state.prediction_results.items()
        ]
        
        selected_key = st.selectbox(
            "Pilih Hasil Prediksi yang akan Ditampilkan",
            options=[key for key, _ in prediction_options],
            format_func=lambda x: next(text for key, text in prediction_options if key == x)
        )
        
        if selected_key:
            result = st.session_state.prediction_results[selected_key]
            model = result['model']
            coefficients = model.coef_
            intercept = model.intercept_
            prediction_year = result['prediction_year']
            city = result['city']
            data_year = result['data_year']

            # 1. Persamaan Regresi
            st.subheader("1. Persamaan Regresi")
            st.markdown(f"**Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃**")
            st.markdown(f"- a (konstanta) = {intercept:.4f}")
            st.markdown(f"- b₁ (koefisien curah hujan) = {coefficients[0]:.4f}")
            st.markdown(f"- b₂ (koefisien suhu) = {coefficients[1]:.4f}")
            st.markdown(f"- b₃ (koefisien luas panen) = {coefficients[2]:.4f}")

            # # 2. Hasil Prediksi
            # st.subheader(f"2. Hasil Prediksi untuk Tahun {prediction_year}")
            # st.markdown(f"**Prediksi hasil panen: {result['prediction']:.3f} ton**")

            # 2. Uji Akurasi Model
            st.subheader("2. Uji Akurasi Model")
            rmse = None
            if city == "Malang":
                rmse = st.session_state.rmse_malang.get(prediction_year)
                eval_data = st.session_state.data_malang.get(prediction_year)
            else:
                rmse = st.session_state.rmse_lumajang.get(prediction_year)
                eval_data = st.session_state.data_lumajang.get(prediction_year)

            if rmse is None and eval_data is not None and not eval_data.empty:
                X_eval = eval_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']]
                y_actual_eval = eval_data['Y']
                y_pred_eval = model.predict(X_eval)
                rmse = np.sqrt(mean_squared_error(y_actual_eval, y_pred_eval))

            if rmse is not None:
                mean_actual_eval = np.mean(eval_data['Y']) if eval_data is not None and not eval_data.empty else 1
                rmse_percentage_eval = (rmse / mean_actual_eval) * 100
                st.markdown(f"### 🎯 RMSE: **{rmse :.4f}**")
                st.success(f"**{rmse_percentage_eval:.2f}%** dari rata-rata aktual")
            else:
                st.info("RMSE tidak tersedia karena data aktual tahun prediksi belum tersedia.")

            # 3. Tabel Perbandingan Data Prediksi
            st.subheader(f"3. Data Prediksi Tahun {prediction_year}")
            if eval_data is not None and not eval_data.empty:
                y_pred_eval = model.predict(eval_data[['X1(CURAH HUJAN)', 'X2(SUHU)', 'X3(LUAS PANEN)']])
                comparison_eval_df = pd.DataFrame({
                    'Bulan': [f"Bulan {i+1}" for i in range(len(eval_data))],
                    'Curah Hujan': eval_data['X1(CURAH HUJAN)'],
                    'Suhu': eval_data['X2(SUHU)'],
                    'Luas Panen': eval_data['X3(LUAS PANEN)'],
                    'Hasil Aktual': eval_data['Y'],
                    'Hasil Prediksi': y_pred_eval,
                    'Selisih': abs(eval_data['Y'] - y_pred_eval)
                })
                st.dataframe(comparison_eval_df)
            else:
                st.info("Data aktual belum tersedia.")

            # 4. Visualisasi Perbandingan
            st.subheader("4. Visualisasi Perbandingan")
            if eval_data is not None and not eval_data.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                months = range(1, len(eval_data) + 1)
                ax.plot(months, eval_data['Y'], 'b-o', label='Aktual')
                ax.plot(months, y_pred_eval, 'r--s', label='Prediksi')
                ax.set_xlabel("Bulan")
                ax.set_ylabel("Hasil Panen (ton)")
                ax.set_title(f"Perbandingan Aktual vs Prediksi ({city}, {prediction_year})")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.info("Visualisasi tidak tersedia karena data aktual belum ada.")

            # 5. Download Hasil
            st.subheader("5. Download Hasil")
            
            # Create download button
            csv_data = []
            csv_data.append("=== INFORMASI MODEL ===")
            csv_data.append(f"Kota,{result['city']}")
            csv_data.append(f"Tahun Prediksi,{result['prediction_year']}")
            csv_data.append(f"Persamaan,Y = {intercept:.4f} + {coefficients[0]:.4f}X₁ + {coefficients[1]:.4f}X₂ + {coefficients[2]:.4f}X₃")
            csv_data.append("")
            if rmse is not None:
                csv_data.append(f"RMSE,{rmse :.4f}")

            csv_data.append("")
            csv_data.append("=== DATA LATIH MODEL ===")
            csv_data.append("Bulan,Curah Hujan,Suhu,Luas Panen,Hasil Aktual")
            for idx, row in result['training_data'].iterrows():
                csv_data.append(f"{idx+1},{row['X1(CURAH HUJAN)']},{row['X2(SUHU)']},{row['X3(LUAS PANEN)']},{row['Y']}")
            csv_data.append("")
            csv_data.append("=== DATA INPUT PREDIKSI ===")
            csv_data.append(f"Curah Hujan,{result['input_data']['curah_hujan']:.2f}")
            csv_data.append(f"Suhu,{result['input_data']['suhu']:.2f}")
            csv_data.append(f"Luas Panen,{result['input_data']['luas_panen']:.3f}")
            csv_data.append(f"Prediksi Hasil Panen,{result['prediction']:.3f} ton")
            csv_data.append("")
            
            csv_data.append("=== DATA PREDIKSI ===")
            csv_data.append("Bulan,Curah Hujan,Suhu,Luas Panen,Hasil Aktual,Hasil Prediksi,Selisih")
            if eval_data is not None and not eval_data.empty:
                for idx, row in eval_data.iterrows():
                    csv_data.append(f"{idx+1},{row['X1(CURAH HUJAN)']},{row['X2(SUHU)']},{row['X3(LUAS PANEN)']},{row['Y']},{y_pred_eval[idx]},{abs(row['Y'] - y_pred_eval[idx])}")
            else:
                csv_data.append("Tidak ada data aktual untuk tahun ini.")
            
            csv_string = "\n".join(csv_data)
            
            st.download_button( 
                label="Download Hasil (CSV)",
                data=csv_string,
                file_name=f"hasil_prediksi_{result['city']}_{result['data_year']}_to_{result['prediction_year']}.csv",
                mime="text/csv"
            )
