# %%
# IMPORT PACKAGE
import sqlalchemy
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import plotly.express as px
import plotly.graph_objects as go

import os.path
import shelve

# %%
# RAW DATA PREPARATION
if os.path.isfile("shelve_apprate_raw.dat"):
	with shelve.open("shelve_apprate_raw") as s:
		object_full = s['object_full']
		dp_full = s['dp_full']
else:
	# DB CONNECTION
	DB_HOST = "root:@127.0.0.1"
	DB_NAME = "co_mcv"
	DB_ENGINE = sqlalchemy.create_engine(f"mysql+pymysql://{DB_HOST}/{DB_NAME}")
	DB_CON = DB_ENGINE.connect()

	# FEATURE LOCATION IN DATABASE
	# OBJECT_FEATURES AND OBJECT_TARGETS
	"""
	object_features = {
		"appraisalid" : "appraisaltb.kodeappraisalTB",
		"jenisobjek" : "appraisal_rincianagunan.jenisagunan", # Convert ke label dengan join
		"luas" : "appraisal_rincianagunan.totluastanah",
		"luasbangunan" : "appraisal_rincianagunan.totluasbangunan",
		"jarak" : "0",
		"bentuk" : "appraisal_keadaantanah.bentuk",
		"lebardepan" : "appraisal_keadaantanah.lebardepan",
		"posisi" : "appraisal_keadaantanah.posisi",
		"kondisi" : "appraisal_keadaantanah.kondisi",
		"peruntukan" : "appraisal_keadaantanah.peruntukan",
		"elevasi" : "appraisal_keadaantanah.elevasi",
		"lebarjalan" : "appraisal_fasilitas.lebarjalan",
		"jenislegal" : "appraisal_legal.jenislegal", # LIMIT 1
		"jumlahlantai" : "appraisalbangunan.jumlahlantai", # LIMIT 1
		"kondisibangunan" : "appraisalbangunan.perawatan" # LIMIT 1

	object_targets = {
		"nilai_pm" : "appraisal_adjdp.pasartanahmkesimpulan" # LIMIT 1
	}

	deviation_features = {
		"nilai_pm_1" : "appraisal_adjdp.pemterendah",
		"nilai_pm_2" : "appraisal_adjdp.pemmedian",
		"nilai_pm_3" : "appraisal_adjdp.pemtertinggi"
	}
	"""

	OBJECT_FULL_SQL = """
							WITH temptable_legal AS (
								SELECT appraisal_legal.kdappraisalTB AS kdappraisalTB, appraisal_legal.jenislegal AS max_jenislegal
								FROM appraisal_legal
								INNER JOIN (SELECT kdappraisalTB, MAX(CAST(luas AS DECIMAL)) AS max_luas FROM appraisal_legal GROUP BY kdappraisalTB) as temp
								ON appraisal_legal.kdappraisalTB = temp.kdappraisalTB
								AND CAST(appraisal_legal.luas AS DECIMAL) = temp.max_luas
								GROUP BY kdappraisalTB
							),
							temptable_bangunan AS (
								SELECT appraisalbangunan.kdappraisalTB, appraisalbangunan.jumlahlantai, appraisalbangunan.perawatan
								FROM appraisalbangunan
								INNER JOIN (SELECT kdappraisalTB, MAX(CAST(jumlahlantai AS INT)) AS max_jumlahlantai FROM appraisalbangunan GROUP BY kdappraisalTB) as temp
								ON appraisalbangunan.kdappraisalTB = temp.kdappraisalTB
								AND CAST(appraisalbangunan.jumlahlantai AS INT) = temp.max_jumlahlantai
								GROUP BY kdappraisalTB
							),
							temptable_dp AS (
								SELECT DISTINCT kdappraisalTB, pemterendah, pemmedian, pemtertinggi, pasartanahmkesimpulan
								FROM appraisal_adjdp
							)
							SELECT
								appraisaltb.kodeappraisalTB AS appraisalid,
								jenisagunan.nama AS jenisobjek,
								appraisal_rincianagunan.totluastanah AS luas,
								appraisal_rincianagunan.totluasbangunan AS luasbangunan,
								0 AS jarak,
								appraisal_keadaantanah.bentuk AS bentuk,
								appraisal_keadaantanah.lebardepan AS lebardepan,
								appraisal_keadaantanah.posisi AS posisi,
								appraisal_keadaantanah.kondisi AS kondisi,
								appraisal_keadaantanah.peruntukan AS peruntukan,
								appraisal_keadaantanah.elevasi AS elevasi,
								appraisal_fasilitas.lebarjalan AS lebarjalan,
								temptable_legal.max_jenislegal AS jenislegal,
								temptable_bangunan.jumlahlantai AS jumlahlantai,
								temptable_bangunan.perawatan AS kondisibangunan,
								temptable_dp.pemterendah AS nilai_pm_1,
								temptable_dp.pemmedian AS nilai_pm_2,
								temptable_dp.pemtertinggi AS nilai_pm_3,
								temptable_dp.pasartanahmkesimpulan AS nilai_pm
							FROM appraisaltb
							LEFT JOIN appraisal_rincianagunan ON appraisaltb.kodeappraisalTB = appraisal_rincianagunan.kdorder
							LEFT JOIN appraisal_keadaantanah ON appraisaltb.kodeappraisalTB = appraisal_keadaantanah.kdappraisalTB
							LEFT JOIN appraisal_fasilitas ON appraisaltb.kodeappraisalTB = appraisal_fasilitas.kdappraisalTB
							LEFT JOIN jenisagunan ON appraisal_rincianagunan.jenisagunan = jenisagunan.kodejenisagunan
							LEFT JOIN temptable_legal ON appraisaltb.kodeappraisalTB = temptable_legal.kdappraisalTB
							LEFT JOIN temptable_bangunan ON appraisaltb.kodeappraisalTB = temptable_bangunan.kdappraisalTB
							LEFT JOIN temptable_dp ON appraisaltb.kodeappraisalTB = temptable_dp.kdappraisalTB
							WHERE appraisaltb.statusorder = '4'

	"""

	object_full = pd.read_sql(OBJECT_FULL_SQL,DB_CON)

	# DP FEATURES AND TARGETS
	"""
	dp_features = {
		"dpid" : "appraisal_dp.kodepembanding",
		"appraisalid" : "appraisal_dp.kdappraisalTB",
		"jenisobjek" : "appraisal_dp.tipeagunan", # Convert ke label dengan join
		"luas" : "appraisal_pembanding_tb.luastanah",
		"luasbangunan" : "appraisal_pembanding_tb.luasbangunan",
		"jarak" : "appraisal_pembanding_tb.jarak", # Perlu Cleansing
		"bentuk" : "appraisal_pembanding_tb.bentuktanah",
		"lebardepan" : "appraisal_pembanding_tb.txtlebardepan",
		"posisi" : "appraisal_pembanding_tb.posisi",
		"kondisi" : "appraisal_pembanding_tb.kondisitanah",
		"peruntukan" : "appraisal_pembanding_tb.txtperuntukan",
		"elevasi" : "appraisal_pembanding_tb.elevasitanah",
		"lebarjalan" : "appraisal_pembanding_tb.lebarjalan", # Perlu Cleansing
		"jenislegal" : "appraisal_pembanding_tb.milik",
		"jumlahlantai" : "appraisal_pembanding_tb.jmllantai",
		"kondisibangunan" : "appraisal_pembanding_tb.kondisibangunan"
	}

	dp_targets = {
		"nilai_pm" : "appraisal_adjdp.indikasinilaitanahmpem",
		"bobot" : "appraisal_adjdp.bobot"
	}
	"""

	DP_FULL_SQL = """
					SELECT
						appraisal_dp.kodepembanding AS dpid,
						appraisal_dp.kdappraisalTB AS appraisalid,
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
						appraisal_adjdp.indikasinilaitanahmpem AS nilai_pm,
						appraisal_adjdp.bobot AS bobot,
						appraisal_adjdp.adjtotal AS adjtotal,
						appraisal_adjdp.adjtotalabs AS adjtotalabs
					FROM appraisal_dp
					LEFT JOIN jenisagunan ON appraisal_dp.tipeagunan = jenisagunan.kodejenisagunan
					LEFT JOIN appraisal_pembanding_tb ON appraisal_dp.kodepembanding = appraisal_pembanding_tb.kodepembanding
					LEFT JOIN appraisal_adjdp ON appraisal_dp.kodepembanding = appraisal_adjdp.kdpem
					LEFT JOIN appraisaltb ON appraisal_dp.kdappraisalTB = appraisaltb.kodeappraisalTB
					WHERE appraisaltb.statusorder = '4' AND appraisal_adjdp.adjtotalabs < '30'
	"""

	dp_full = pd.read_sql(DP_FULL_SQL,DB_CON)

	# DB CLOSE CONNECTION
	DB_CON.close()
	DB_ENGINE.dispose()

	# SAVE TO SHELVE
	with shelve.open("shelve_apprate_raw") as s:
		s["object_full"] = object_full
		s["dp_full"] = dp_full

print(object_full.info())
print(dp_full.info())

# DATA CLEANSING
# %%
# Set Index and drop duplicate ID
object_full.set_index("appraisalid",drop=True,inplace=True)
object_full = object_full[~object_full.index.duplicated(keep='last')]
dp_full.set_index("dpid",drop=True,inplace=True)
dp_full = dp_full[~dp_full.index.duplicated(keep='last')]
dp_full.set_index(["appraisalid"],append=True,drop=True,inplace=True)

# %%
# Fill NA AND DROP (luasbangunan, jumlahlantai, kondisibangunan)
def fill_drop_full(asset_full):
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

object_full = fill_drop_full(object_full)
dp_full = fill_drop_full(dp_full)

# Drop object that has no exact 3 dp pairing
count_values_appraisalid = dp_full.index.get_level_values('appraisalid').to_series().value_counts()
appraisalid_with_3_dp = count_values_appraisalid[count_values_appraisalid == 3].index.to_series()

# Drop the ID that has no pair between object and dp
object_full = object_full[object_full.index.to_series().isin(dp_full.index.unique(level='appraisalid').tolist()) &\
							object_full.index.to_series().isin(appraisalid_with_3_dp)]
dp_full = dp_full[dp_full.index.isin(object_full.index.tolist(), level='appraisalid') &\
					dp_full.index.isin(appraisalid_with_3_dp, level='appraisalid')]

print(object_full.info())
print(dp_full.info())

# %%
# Category to Numeric Typing cleansing
dp_jarak_mapping = {
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

dp_lebarjalan_mapping = {
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

dp_full['jarak'] = dp_full['jarak'].map(dp_jarak_mapping)
dp_full['lebarjalan'] = dp_full['lebarjalan'].map(dp_lebarjalan_mapping)
object_full['lebarjalan'].replace(r'\D+','',regex=True,inplace=True)

# %%
# Assign Column Type
object_full[['luas','luasbangunan','lebardepan','lebarjalan','jumlahlantai','nilai_pm_1','nilai_pm_2','nilai_pm_3','nilai_pm']] = \
	object_full[['luas','luasbangunan','lebardepan','lebarjalan','jumlahlantai','nilai_pm_1','nilai_pm_2','nilai_pm_3','nilai_pm']]\
	.apply(pd.to_numeric)

dp_full[['luas','luasbangunan','jumlahlantai','nilai_pm','adjtotal','adjtotalabs']] =\
	dp_full[['luas','luasbangunan','jumlahlantai','nilai_pm','adjtotal','adjtotalabs']]\
	.apply(pd.to_numeric)

# Drop Nilai Per Meter2 = 0
object_full = object_full[(object_full['nilai_pm'] > 0) &\
							(object_full['luas'] > 0) &\
							(object_full['lebardepan'] > 0) &\
							(object_full['lebarjalan'] > 0)]
dp_full = dp_full[(dp_full['nilai_pm'] > 0) &\
					(dp_full['luas'] > 0) &\
					(dp_full['jarak'] > 0) &\
					(dp_full['lebardepan'] > 0) &\
					(dp_full['lebarjalan'] > 0)]

# Drop object that has no exact 3 dp pairing
count_values_appraisalid = dp_full.index.get_level_values('appraisalid').to_series().value_counts()
appraisalid_with_3_dp = count_values_appraisalid[count_values_appraisalid == 3].index.to_series()

# Drop the ID that has no pair between object and dp
object_full = object_full[object_full.index.to_series().isin(dp_full.index.unique(level='appraisalid').tolist()) &\
							object_full.index.to_series().isin(appraisalid_with_3_dp)]
dp_full = dp_full[dp_full.index.isin(object_full.index.tolist(), level='appraisalid') &\
					dp_full.index.isin(appraisalid_with_3_dp, level='appraisalid')]
# %%
# Take Out Ruko sebagai data
object_nonruko = object_full[object_full.jenisobjek != 'Ruko'].copy()
dp_nonruko = dp_full[dp_full.index.isin(object_nonruko.index.tolist(), level='appraisalid')].copy()

# %%
# Numeric Feature Engineering: Features Interaction: luastanah/lebardepan, ruko: luasbangunan/jumlahlantai
object_nonruko['depth'] = object_nonruko['luas']/object_nonruko['lebardepan']
dp_nonruko['depth'] = dp_nonruko['luas']/dp_nonruko['lebardepan']


# %%
# Targets - Features (Numerical, Category Ordinal, Category Nominal) Splitting
"""
object_full.columns = ['jenisobjek', 'luas', 'luasbangunan', 'jarak', 'bentuk', 'lebardepan',
	'posisi', 'kondisi', 'peruntukan', 'elevasi', 'lebarjalan',
	'jenislegal', 'jumlahlantai', 'kondisibangunan', 'nilai_pm_1',
	'nilai_pm_2', 'nilai_pm_3', 'nilai_pm'
]

dp_full.columns = ['jenisobjek', 'luas', 'luasbangunan', 'jarak', 'bentuk', 'lebardepan',
	'posisi', 'kondisi', 'peruntukan', 'elevasi', 'lebarjalan',
	'jenislegal', 'jumlahlantai', 'kondisibangunan', 'nilai_pm', 'bobot'
]
"""

features_all = ['jenisobjek', 'luas', 'luasbangunan', 'jarak', 'bentuk', 'lebardepan',
	'posisi', 'kondisi', 'peruntukan', 'elevasi', 'lebarjalan',
	'jenislegal', 'jumlahlantai', 'kondisibangunan']

# features_numerical = ['luas','jarak','lebardepan','lebarjalan']
features_numerical = ['luas','lebardepan','lebarjalan','depth','jarak']
features_ordinal = []
features_nominal = ['bentuk','posisi','kondisi','peruntukan','elevasi','jenislegal','jenisobjek']

features_numerical_ruko = ['luasbangunan','jumlahlantai']
features_ordinal_ruko = ['kondisibangunan']

features_nonruko = features_numerical+features_ordinal+features_nominal

object_nonruko_features = object_nonruko[features_nonruko].copy()
dp_nonruko_features = dp_nonruko[features_nonruko].copy()

# %%
# CATEGORICAL FEATURES: Nominal encoding by dummies
object_nonruko_features_ohe = pd.get_dummies(object_nonruko_features[features_nominal], prefix=features_nominal)
dp_nonruko_features_ohe = pd.get_dummies(dp_nonruko_features[features_nominal], prefix=features_nominal)

object_nonruko_features = object_nonruko_features.drop(features_nominal, axis=1)
dp_nonruko_features = dp_nonruko_features.drop(features_nominal, axis=1)

object_nonruko_features = object_nonruko_features.join(object_nonruko_features_ohe)
dp_nonruko_features = dp_nonruko_features.join(dp_nonruko_features_ohe)

# Add column that has no pair in OHE
object_nonruko_features[dp_nonruko_features.columns.difference(object_nonruko_features.columns)] = 0
dp_nonruko_features[object_nonruko_features.columns.difference(dp_nonruko_features.columns)] = 0

# %%
# Cosine Similarity
# A = object_nonruko_features
# B = dp_nonruko_features

def calc_cosine_similarity(a,b):
	# Dot Product
	a_dot_b = (a*b).sum(axis=1)
	# Norm
	norm_a = a.apply(np.linalg.norm, axis=1)
	norm_b = b.apply(np.linalg.norm, axis=1)
	# Cos Teta
	cosine_similarity = pd.DataFrame(a_dot_b/(norm_a*norm_b),columns=['cosine_similarity'])
	return cosine_similarity

# def calc_adj_cosine_similarity(a,b):
# 	max_ab = np.maximum(a,b.max())
# 	min_ab = np.minimum(a,b.min())
# 	max_ab = max_ab[features_numerical+features_ordinal]
# 	min_ab = min_ab[features_numerical+features_ordinal]
# 	normalized_a = ((a[features_numerical+features_ordinal]-min_ab)/(max_ab-min_ab)).fillna(value=0).replace([np.inf,-np.inf],0)
# 	normalized_a = normalized_a.join(a[a.columns.difference(features_numerical+features_ordinal)])
# 	normalized_b = ((b[features_numerical+features_ordinal]-min_ab)/(max_ab-min_ab)).fillna(value=0).replace([np.inf,-np.inf],0)
# 	normalized_b = normalized_b.join(b[b.columns.difference(features_numerical+features_ordinal)])
# 	cosine_similarity = calc_cosine_similarity(normalized_a,normalized_b.swaplevel(0,1))
# 	return cosine_similarity

def calc_adj_cosine_similarity(a,b):
	mean_ab = (a+b)/2
	mean_ab = mean_ab[features_numerical+features_ordinal]
	unbiased_stdev_ab = np.sqrt((np.square((a-mean_ab))+np.square((b-mean_ab)))/1)
	unbiased_stdev_ab = unbiased_stdev_ab[features_numerical+features_ordinal]

	standardized_a = ((a[features_numerical+features_ordinal]-mean_ab)/unbiased_stdev_ab).fillna(value=0)
	standardized_a = standardized_a.join(a[a.columns.difference(features_numerical+features_ordinal)])
	standardized_b = ((b[features_numerical+features_ordinal]-mean_ab)/unbiased_stdev_ab).fillna(value=0)
	standardized_b = standardized_b.join(a[a.columns.difference(features_numerical+features_ordinal)])

	cosine_similarity = calc_cosine_similarity(standardized_a,standardized_b.swaplevel(0,1))
	return cosine_similarity

a = object_nonruko_features
b = dp_nonruko_features

cosine_similarity = calc_adj_cosine_similarity(a,b)
cosine_similarity = cosine_similarity.swaplevel(0,1)


# %%
# Hitung bobot
cosine_similarity['weight'] = cosine_similarity['cosine_similarity']/cosine_similarity.groupby('appraisalid')['cosine_similarity'].sum()

# %%
# Hitung Nilai
cosine_similarity = cosine_similarity.join(dp_nonruko['nilai_pm'])
cosine_similarity['weighted_value'] = cosine_similarity['weight']*cosine_similarity['nilai_pm']
market_value = pd.concat([cosine_similarity.groupby('appraisalid')['weighted_value'].sum().rename('prediction'),\
				cosine_similarity.groupby('appraisalid')['cosine_similarity'].sum().rename('sum_similarity')],\
				axis=1)

# %%
# Perbandingan dengan real (MAE, RSME, R2)
market_value = market_value.join(object_nonruko['nilai_pm'].rename('original'))
market_value['deviation'] = np.abs(market_value['original']-market_value['prediction'])/market_value['original']

# Drop Anomaly and to be rejected if sum similarity <=1.5
# market_value = market_value[market_value['deviation']<1]
market_value = market_value[market_value['sum_similarity']>1.5]

market_value['mae'] = metrics.mean_absolute_error(market_value['original'],market_value['prediction'])
market_value['rmse'] = np.sqrt(metrics.mean_squared_error(market_value['original'],market_value['prediction']))
market_value['mape'] = metrics.mean_absolute_percentage_error(market_value['original'],market_value['prediction'])
market_value['r2'] = metrics.r2_score(market_value['original'],market_value['prediction'])

# %%
# Graph Sum Similarity vs deviation
market_value_filtered = market_value
fig = px.scatter(x=market_value_filtered['sum_similarity'],\
				y=market_value_filtered['deviation'],\
				labels={'x':'sum_similarity','y':'deviation_ori_to_pred'},\
				trendline="ols",trendline_color_override='red',\
				marginal_x='histogram', marginal_y='histogram')
fig.show()

# Correl
market_value_filtered[['sum_similarity','deviation']].corr()

# Cosine Similarity
cosine_similarity_filtered = cosine_similarity
fig = px.histogram(cosine_similarity['cosine_similarity'],x='cosine_similarity')
fig.show()
fig = px.histogram(cosine_similarity['weight'],x='weight')
fig.show()
# %%
# Simple Average
object_nonruko_dropped = object_nonruko.loc[object_nonruko_features.index,['nilai_pm_1','nilai_pm_2','nilai_pm_3']]
simple_market_value = pd.DataFrame(object_nonruko_dropped.mean(axis=1), columns=['prediction'])
simple_market_value['original'] = object_nonruko['nilai_pm']
simple_market_value['deviation'] = np.abs(simple_market_value['original']-simple_market_value['prediction'])/simple_market_value['original']
simple_market_value = simple_market_value[simple_market_value['deviation']<1]
simple_market_value['mae'] = metrics.mean_absolute_error(simple_market_value['original'],simple_market_value['prediction'])
simple_market_value['rmse'] = np.sqrt(metrics.mean_squared_error(simple_market_value['original'],simple_market_value['prediction']))
simple_market_value['mape'] = metrics.mean_absolute_percentage_error(simple_market_value['original'],simple_market_value['prediction'])
simple_market_value['r2'] = metrics.r2_score(simple_market_value['original'],simple_market_value['prediction'])

# %%
# Adjusted Cosine SImilarity
# MAE, RSME, R2

# Feature Selection
# %%
