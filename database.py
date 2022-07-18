from pydantic import BaseModel
from decimal import Decimal
from typing import Any
from enum import Enum

import sqlalchemy
import numpy as np
import pandas as pd

class AssetAppraised(BaseModel):
	kodejenisagunan: int
	kecamatan: str
	kota: str
	luas: float
	luasbangunan: float
	bentuk: str
	lebardepan: float
	posisi: str
	kondisi: str
	peruntukan: str
	elevasi: str
	lebarjalan: float
	jenislegal: str
	jumlahlantai: int
	kondisibangunan: str
	
def obj_to_df(obj):
	df = pd.DataFrame(obj).transpose()
	df.columns = df.iloc[0]
	df = df.drop(0).reset_index(drop=True)
	return df

def calc_cosine_similarity(a,b):
	# Dot Product
	a_dot_b = (a.iloc[0]*b).sum(axis=1)
	# Norm
	norm_a = a.apply(np.linalg.norm, axis=1)
	norm_b = b.apply(np.linalg.norm, axis=1)
	# Cos Teta
	cosine_similarity = pd.DataFrame(a_dot_b/(norm_a.iloc[0]*norm_b),columns=['cosine_similarity'])
	return cosine_similarity

def calc_adj_cosine_similarity(asset_appraised_features,dp_features,features_numerical,features_ordinal):
	mean_dp = dp_features.mean()
	mean_dp = mean_dp[features_numerical+features_ordinal]
	unbiased_stdev_dp = dp_features.std()
	unbiased_stdev_dp = unbiased_stdev_dp[features_numerical+features_ordinal]

	standardized_a = ((asset_appraised_features[features_numerical+features_ordinal]-mean_dp)/unbiased_stdev_dp).fillna(value=0)
	standardized_a = standardized_a.join(asset_appraised_features[asset_appraised_features.columns.difference(features_numerical+features_ordinal)])
	standardized_b = ((dp_features[features_numerical+features_ordinal]-mean_dp)/unbiased_stdev_dp).fillna(value=0)
	standardized_b = standardized_b.join(dp_features[dp_features.columns.difference(features_numerical+features_ordinal)])

	cosine_similarity = calc_cosine_similarity(standardized_a,standardized_b)
	return cosine_similarity

# ==========
# CONSTANT
# ==========
# N Market Data To Be Combined
N_MARKET_DATA = 5

# Category to Numeric Typing cleansing
JARAK_MAPPING = {
	"-/+ 50 m" : 50,
	"-/+ 100 m" : 100,
	"-/+ 150 m" : 150,
	"-/+ 200 m" : 200,
	"-/+ 250 m" : 250,
	"-/+ 300 m" : 300,
	"-/+ 400 m" : 400,
	"-/+ 500 m" : 500,
	"-/+ 1 km" : 1000,
	"-/+ 2 km" : 2000,
	"-/+ 3 km" : 3000,
	"-/+ 4 km" : 4000,
	"-/+ 5 km" : 5000,
	">5 km" : 10000
}

LEBARJALAN_MAPPING = {
	"-/+ 2 m" : 2,
	"-/+ 2.5 m" : 2.5,
	"-/+ 3 m" : 3,
	"-/+ 3.5 m" : 3.5,
	"-/+ 4 m" : 4,
	"-/+ 4.5 m" : 4.5,
	"-/+ 5 m" : 5,
	"-/+ 5.5 m" : 5.5,
	"-/+ 6 m" : 6,
	"-/+ 6.5 m" : 6.5,
	"-/+ 7 m" : 7,
	"-/+ 7.5 m" : 7.5,
	"-/+ 8 m" : 8,
	"-/+ 9 m" : 9,
	"-/+ 10 m" : 10,
	"> 10 m" : 20,
}

"""
features = ['jenisobjek', 'luas', 'luasbangunan', 'jarak', 'bentuk', 'lebardepan',
	'posisi', 'kondisi', 'peruntukan', 'elevasi', 'lebarjalan',
	'jenislegal', 'jumlahlantai', 'kondisibangunan']
"""

# Feature Selection
features_numerical = ['luas','lebardepan','lebarjalan','depth','luasbangunan','jumlahlantai','building_land_ratio']
features_ordinal = []
features_nominal = ['bentuk','posisi','kondisi','peruntukan','elevasi','jenislegal','kondisibangunan']
features_all = features_numerical+features_ordinal+features_nominal

# ==========

def connectDB():
	DB_HOST = "root:@127.0.0.1"
	DB_NAME = "co_mcap"
	DB_ENGINE = sqlalchemy.create_engine(f"mysql+pymysql://{DB_HOST}/{DB_NAME}")
	DB_CON = DB_ENGINE.connect()
	return DB_ENGINE, DB_CON

class FullAppraisal:
	def __init__(self, asset_appraised:AssetAppraised):
		# ==========
		# ASSET APPRAISED
		# ==========
		# AssetAppraised Object to Dataframe
		asset_appraised_df = obj_to_df(asset_appraised)
		
		# Feature Engineering
		asset_appraised_features = self.add_features_engineering(asset_appraised_df)
		
		# ==========
		# MARKET DATA
		# ==========
		# Get Filtered Market Data
		dp_full = self.get_dp_filtered(asset_appraised)
		print(dp_full.info())

		# Market Data Cleansing
		dp_cleansed = self.data_cleansing(dp_full)

		# Feature Engineering
		dp_features = self.add_features_engineering(dp_cleansed)

		asset_appraised_features, dp_features = self.add_blank_col_ohe(asset_appraised_features,dp_features)
		
		# ==========
		# Cosine Similarity
		# ==========
		cosine_similarity = calc_adj_cosine_similarity(
			asset_appraised_features,dp_features,
			features_numerical,features_ordinal
		)

		# Get Top N Market Data
		top_cosine_dp_id = cosine_similarity.nlargest(n=N_MARKET_DATA,columns='cosine_similarity')

		# ==========
		# Market Value Calculation
		# ==========
		# Get Market Data Details
		self.top_market_data = dp_full.loc[dp_full.index.isin(top_cosine_dp_id.index.get_level_values(0).to_list())]
		# Cosine Similarity
		self.top_market_data['cosine_similarity'] = top_cosine_dp_id['cosine_similarity']
		# Weight
		self.top_market_data['weight'] = top_cosine_dp_id['cosine_similarity']/top_cosine_dp_id['cosine_similarity'].sum()
		# Calculation
		self.asset_appraised_market_value = (self.top_market_data['weight']*pd.to_numeric(self.top_market_data['nilai_pm'])).sum()
	
	def get_dp_filtered(self,asset_appraised):
		DB_ENGINE, DB_CON = connectDB()
		DP_FULL_SQL = f"""
					SELECT
						appraisal_dp.kodepembanding AS dpid,
						appraisal_dp.tipeagunan AS kodejenisagunan,
						jenisagunan.nama AS jenisobjek,
						appraisal_pembanding_tb.luastanah AS luas,
						appraisal_pembanding_tb.luasbangunan AS luasbangunan,
						appraisal_pembanding_tb.jarak AS jarak,
						appraisal_pembanding_tb.bentuktanah AS bentuk,
						appraisal_pembanding_tb.txtlebardepan AS lebardepan,
						appraisal_pembanding_tb.posisi AS posisi,
						appraisal_pembanding_tb.kondisitanah AS kondisi,
						appraisal_pembanding_tb.txtperuntukan AS peruntukan,
						appraisal_pembanding_tb.elevasitanah AS elevasi,
						appraisal_pembanding_tb.lebarjalan AS lebarjalan,
						appraisal_pembanding_tb.milik AS jenislegal,
						appraisal_pembanding_tb.jmllantai AS jumlahlantai,
						appraisal_pembanding_tb.kondisibangunan AS kondisibangunan,
						appraisal_pembanding_tb.desa AS desa,
						appraisal_pembanding_tb.kecamatan AS kecamatan,
						appraisal_pembanding_tb.kota AS kota,
						appraisal_pembanding_tb.provinsi AS provinsi,
						appraisal_adjdp.indikasinilaitanahmpem AS nilai_pm
					FROM appraisal_dp
					LEFT JOIN jenisagunan ON appraisal_dp.tipeagunan = jenisagunan.kodejenisagunan
					LEFT JOIN appraisal_pembanding_tb ON appraisal_dp.kodepembanding = appraisal_pembanding_tb.kodepembanding
					LEFT JOIN appraisal_adjdp ON appraisal_dp.kodepembanding = appraisal_adjdp.kdpem
					LEFT JOIN appraisaltb ON appraisal_dp.kdappraisalTB = appraisaltb.kodeappraisalTB
					WHERE
						appraisal_dp.tipeagunan = '{asset_appraised.kodejenisagunan}'
						AND
						appraisal_pembanding_tb.kecamatan = '{asset_appraised.kecamatan}'
						AND
						appraisal_pembanding_tb.kota = '{asset_appraised.kota}'
		"""
		dp_full = pd.read_sql(DP_FULL_SQL,DB_CON)
		DB_CON.close()
		DB_ENGINE.dispose()
		return dp_full
	
	def fill_drop_full(self,asset_full):
		asset_full.replace(r'^\s*$',np.nan,regex=True,inplace=True)
		asset_full.replace('',np.nan,inplace=True)
		asset_full.replace('NaN',np.nan,inplace=True)
		asset_full.replace(',','.',regex=True,inplace=True)
		asset_full.replace(r'^-*$',0,regex=True,inplace=True)
		asset_full['lebardepan'].replace(r'\D+','',regex=True,inplace=True)
		asset_full['jumlahlantai'].replace(r'\D+','',regex=True,inplace=True)
		asset_full['luasbangunan'].fillna(value=0,inplace=True)
		asset_full['jumlahlantai'].fillna(value=0,inplace=True)
		asset_full['kondisibangunan'].fillna(value='Baik', inplace=True)
		asset_full.dropna(axis=0, inplace=True)
		return asset_full

	def data_cleansing(self,dp_full):
		# Set Index and drop duplicate ID
		dp_full.set_index("dpid",drop=True,inplace=True)
		dp_full = dp_full[~dp_full.index.duplicated(keep='last')]
		
		# Fill NA AND DROP (luasbangunan, jumlahlantai, kondisibangunan)
		dp_full = self.fill_drop_full(dp_full)

		# Numerical Mapping
		dp_full['jarak'] = dp_full['jarak'].map(JARAK_MAPPING)
		dp_full['lebarjalan'] = dp_full['lebarjalan'].map(LEBARJALAN_MAPPING)

		# Assign Column Type
		dp_full[['luas','luasbangunan','jumlahlantai','nilai_pm']] =\
			dp_full[['luas','luasbangunan','jumlahlantai','nilai_pm']]\
			.apply(pd.to_numeric)
		
		# Drop Nilai Per Meter2 = 0
		dp_full = dp_full[(dp_full['nilai_pm'] > 0) &\
							(dp_full['luas'] > 0) &\
							(dp_full['jarak'] > 0) &\
							(dp_full['lebardepan'] > 0) &\
							(dp_full['lebarjalan'] > 0)]

		return dp_full
	
	def add_features_engineering(self,dp_cleansed):
		# Numeric Feature Engineering: Features Interaction: luastanah/lebardepan, ruko: luasbangunan/luastanah
		dp_cleansed['depth'] = dp_cleansed['luas']/dp_cleansed['lebardepan']
		dp_cleansed['building_land_ratio'] = dp_cleansed['luasbangunan']/dp_cleansed['luas']
		dp_cleansed_features = dp_cleansed[features_all].copy()

		# CATEGORICAL FEATURES: Nominal encoding by dummies
		dp_cleansed_features_ohe = pd.get_dummies(dp_cleansed_features[features_nominal], prefix=features_nominal)
		dp_cleansed_features = dp_cleansed_features.drop(features_nominal, axis=1)
		dp_cleansed_features = dp_cleansed_features.join(dp_cleansed_features_ohe)

		return dp_cleansed_features

	def add_blank_col_ohe(self,asset_appraised_features,dp_features):
		# Add column that has no pair in OHE
		asset_appraised_features[dp_features.columns.difference(asset_appraised_features.columns)] = 0
		dp_features[asset_appraised_features.columns.difference(dp_features.columns)] = 0

		return asset_appraised_features, dp_features

if __name__ == "__main__":
	asset_appraised_example = AssetAppraised(
		kodejenisagunan = 1,
		kecamatan = "Nabire",
		kota = "Nabire",
		luas = 150,
		luasbangunan = 100,
		bentuk = "Persegi Panjang",
		lebardepan = 7,
		posisi = "Tengah",
		kondisi = "Tanah Matang/KSB",
		peruntukan = "Pemukiman",
		elevasi = "Sejajar Jalan",
		lebarjalan = 5,
		jenislegal = "SHM",
		jumlahlantai = 1,
		kondisibangunan = "Baik"
	)

	full_appraisal = FullAppraisal(asset_appraised_example)

	print("Done")