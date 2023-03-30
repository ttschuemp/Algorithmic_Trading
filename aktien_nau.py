from base64 import b64encode
import requests
import pandas as pd
import json
import config_msci_api as config
import yaml
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
from datetime import timedelta


# config params
path_zert = config.path_zert

# esg API
api_key = config.api_esg_key
api_secret = config.api_esg_secret

# index API
api_index_key = config.api_index_key
api_index_secret = config.api_index_secret

b64login = b64encode(bytes('%s:%s' % (api_key, api_secret), encoding='utf-8')).decode()
b64login_index = b64encode(bytes('%s:%s' % (api_index_key, api_index_secret), encoding='utf-8')).decode()

# yaml factor list
with open("esg exclusion list and nau scores.yml", 'r') as stream:
    factor_list = yaml.safe_load(stream)


acwi_imi_index = 664204
index_code = acwi_imi_index


# add calc date and as of date in yaml file and integrate it in url string

# calc_date = factor_list['rebalancing_calc_date']
# as_of_date = factor_list['rebalancing_as_of_date']

calc_date = 20230329
as_of_date = 20230329

# functions to get index form index API
url_index = "https://api.msci.com/index/constituents/v1.0/indexes/" + str(index_code) + "/initial?calc_date=" + str(
    calc_date) + "&as_of_date=" + str(as_of_date) + \
            "&output=SECURITY_IDENTIFIERS"

headers = {'content-type': "application/x-www-form-urlencoded",
           'accept': "application/json", 'accept-encoding': "deflate,gzip",
           "Authorization": "Basic %s" % b64login_index}
response = requests.request("GET", url_index, headers=headers, verify=path_zert)
data_index = response.json()
results_index_full = data_index['indexConstituents'][0]['SECURITY_IDENTIFIERS']
df_index = pd.json_normalize(results_index_full).dropna(subset=['isin'])

df_isin = df_index['isin'].to_list()


# df_index[df_index.apply(lambda row: row.astype(str).str.contains("Deletion", case=False).any(), axis=1)]

# function to get ESG data from ESG API

url = "https://api.msci.com/esg/data/v1.0/issuers?"
headers = { "Authorization": "Basic %s" % b64login,
            'content-type': "application/json" }

# ESG factor list BLKB exclusion criteria
factor_list_esg_exclusions = factor_list['exclusion criteria']
parent_child = factor_list['parent rating']

payload = json.dumps({"issuer_identifier_list": df_isin,
                      "factor_name_list": factor_list_esg_exclusions,
                      "parent_child": parent_child
                      })

response = requests.request("POST", url, data=payload, headers=headers, verify=path_zert)
data = response.json()
results = data['result']['issuers']
index_result = pd.DataFrame.from_dict(pd.json_normalize(results), orient='columns')

keyorder = factor_list_esg_exclusions
# important to know: with reindexing columns from keyorder, you only take those columns out that are specified in the yml-data "factor_list_esg_exclusion"
index_data_esg = index_result.reindex(columns=keyorder)

# for dual-class share comps (e.g., Alphabet), ISSUER_ISIN captures only one of them (i.e., same ISIN for both Alphabet shares)
# we take the CLIENT_IDENTIFIER, which captures dual-class shares
index_data_esg.drop(columns=["ISSUER_ISIN"], inplace=True)
index_data_esg.rename(columns={"CLIENT_IDENTIFIER":"ISSUER_ISIN" }, inplace=True)

# getting MSCI security codes (i.e., identifiers) to the dataframes in order to map GICS Sub industries to it
df_msci_sec_code = df_index[['msci_security_code', "isin"]]
df_msci_sec_code = df_msci_sec_code.rename(columns={"isin":"ISSUER_ISIN"})

index_data_esg = pd.merge(index_data_esg, df_msci_sec_code, on="ISSUER_ISIN", how="inner")

#getting MSCI GICS Sub industries for each constituent (world) to the dataframes

headers = {'content-type': "application/x-www-form-urlencoded",
           'accept': "application/json", 'accept-encoding': "deflate,gzip",
           "Authorization": "Basic %s" % b64login_index}

params = {'calc_date': calc_date, 'as_of_date': as_of_date,'output':'GICS','distribution_zone':'WORLD','cumulative':'true'}
r = requests.get('https://api.msci.com/index/secmaster/security/v2.0/securities/ALL/marketopen', headers=headers, params=params, verify=path_zert)
df_subsecs=pd.DataFrame.from_dict(pd.json_normalize(r.json()['securities']), orient='columns')
df_subsecs=df_subsecs[['msci_security_code','GICS.sub_industry']]
df_subsecs['GICS.sub_industry'] = df_subsecs['GICS.sub_industry'].astype('str')

index_data_esg = pd.merge(index_data_esg, df_subsecs, on="msci_security_code", how="inner")

#get GIC

# check double-share
# index_result[index_result['ISSUER_NAME'].str.contains('ALPHABET')]

## ESG exclusion
# exclusion of GICS Sub industry ["Airlines", "Aerospace & Defense", "Electric Utilities"] via codes and also names (double-check if one or the other changes)
# why mapping needed and not going directly via GICS_SUB_IND?
# -> API gives only coverage for additional information (like GICS Sub industry) for those that have a rating available. Otherwise it is stated as "None"
gics_sub_exclusion = ["20302010", "20101010", "55101010"]
gics_sub_exclusion2 = ["Airlines", "Aerospace & Defense ", "Electric Utilities", "Passenger Airlines"]
data_filter = index_data_esg.copy()
data_filter = data_filter[~data_filter['GICS.sub_industry'].isin(gics_sub_exclusion)]
data_filter = data_filter[~data_filter['GICS_SUB_IND'].isin(gics_sub_exclusion2)]


# fill nan weapon revenue percentage with 0
weapon_exclusion = ['GAM_OPER_MAX_REV_PCT', 'WEAP_MAX_REV_PCT',
                    'FIREARM_PROD_MAX_REV_PCT', 'FIREARM_RET_MAX_REV_PCT']

data_filter[weapon_exclusion] = data_filter[weapon_exclusion].fillna(0)

# exclusion of weapon producer with revenue share > 5%
data_filter = data_filter[(data_filter['GAM_OPER_MAX_REV_PCT'] < 5) & (data_filter['WEAP_MAX_REV_PCT'] < 5)
                          & (data_filter['FIREARM_PROD_MAX_REV_PCT'] < 5) & (data_filter['FIREARM_RET_MAX_REV_PCT'] < 5)]

# BLKB exclusion criteria
esg_exclusion_criterias = ['TOB_PRODUCER','NUC_UTILITY','NUC_URANIUM_MINE','INDUSTRY_ENERGY_PRODUCER',
                           'INDUSTRY_ENERGY_APPLICATION','COAL_RESERVES','GMO_AGRICULTURE','WEAP_GPRODUCER',
                           'AE_PRODUCER','CB_MANUFACTURER', 'LM_MANUFACTURER', 'WEAP_BIO_CHEM_SYSTEM', 'DU_WEAP',
                           'WEAP_NUC_SYSTEM']
# change None to False
data_filter.fillna(False, inplace = True)
#data_filter.dtypes

# remove all rows containing BLKB ESG exclusion criterias
data_filter = data_filter[~data_filter[esg_exclusion_criterias].isin([True]).any(axis=1)]

#drop duplicates
data_filter.drop_duplicates(inplace=True)

#manually deleting some  comps compared to excel-version (Airlines etc)
#data_filter = data_filter[~data_filter['ISSUERID'].isin(["IID000000002156564","IID000000002156468","IID000000002162579","IID000000002162579","IID000000002162579","IID000000002137927","IID000000002187724","IID000000002404389","IID000000002811565","IID000000002647080","IID000000002675793",])]


#%%
data_filter = data_filter.sort_values("INDUSTRY_ADJUSTED_SCORE", ascending=False).reset_index(drop=True)
data_filter.rename(columns={"ISSUER_ISIN":"ISIN", "ISSUER_NAME":"Name", "INDUSTRY_ADJUSTED_SCORE":"Score", "IVA_COMPANY_RATING":"Rating", "ISSUERID":"MSCI ID" }, inplace=True)
data_filter = data_filter[["Name", "MSCI ID", "ISIN", "GICS_SUB_IND", "Score", "Rating", "msci_security_code"]]
data_filter["Score"] = data_filter["Score"].astype(float)

#manually adding some comps that don't have a ESG rating via API compared to ESG manager

#index_data_esg1 = index_data_esg[index_data_esg['ISSUERID'].isin(["IID000000002928741",   "IID000000002161103",  "IID000000002158175",  "IID000000005076065",  "IID000000002133327",  "IID000000002137779",  "IID000000002126580",  "IID000000002398099"   ,"IID000000002453229", "IID000000002186954",  "IID000000002186456",  "IID000000002176636",  "IID000000002166953"   ,"IID000000002700476"  ,"IID000000002158490"  "IID000000002187175"   ,"IID000000002140894"  ,"IID000000002149126"  ,"IID000000002124790", "IID000000002149076",  "IID000000002833388",  "IID000000002133085",  "IID000000002132507"])]
#data_filter = pd.concat([index_data_esg1, data_filter], ignore_index=True)


#create top 45 and 49

threshold_49 = np.floor(len(data_filter)*0.49).astype("int32")
threshold_rating_49 = data_filter.iloc[threshold_49,:]["Score"]

threshold_45 = np.floor(len(data_filter)*0.45).astype("int32")
threshold_rating_45 = data_filter.iloc[threshold_45,:]["Score"]

top_49= data_filter.loc[(data_filter['Score'] >= threshold_rating_49)]
top_45= data_filter.loc[(data_filter['Score'] >= threshold_rating_45)]


between_4= top_49.loc[(top_49['Score'] < threshold_rating_45)]


#%%

NAU_T_1 = pd.read_excel(r'Q:\PBIS_IC_alle_MA\Nachhaltigkeit (SRI)\Universum Aktien\BLKB NAU Aktien\Aktien_NAU_20230228.xlsx', sheet_name='BoU')
NAU_T_1 = NAU_T_1[["Score", "Rating", "MSCI ID", "Name", "ISIN", "Region", "Sector","Auf der Kippe?", "Seit wann"]]
NAU_T_1['Seit wann'] = pd.to_datetime(NAU_T_1['Seit wann'])

#adjusting rating of missing allianz double-share
allianz_rating = NAU_T_1[NAU_T_1["ISIN"] == "DE0008404005" ]["Score"].reset_index(drop=True)[0]
NAU_T_1.loc[NAU_T_1['ISIN'].str.contains('DE000A3H23N4'), 'Score'] = allianz_rating

#seperating stocks from list that are overruled or under review; as well as allianz
overruled = NAU_T_1[NAU_T_1["Score"].str.contains("OVERRULED|REVIEW", na=False) | NAU_T_1["ISIN"].str.contains("DE000A3H23N4", na=False)]
NAU_T_1 = NAU_T_1.iloc[:-len(overruled.index),:]

NAU_T_1["Score"] = NAU_T_1["Score"].astype(float)

#changing dates of grenzwertig to end of month instead of beginning of month (later considered for deletion of expiring grenzwertig constituents)

NAU_T_1['Seit wann'] = NAU_T_1['Seit wann'] + MonthEnd(0)

#%%
#check if stocks in "between_4" should be stated "Grenzwertig"
# note that once a stock falls below top_49, it won't be "Grenzwertig" anymore.
# So we check if current "between_4" stocks were before (NAU T-1) either in top_45 or were stated as "Grenzwertig"
# creating column "_merge" in between_4 which compares existence to NAU_T_1 with "left_only" (doesn't exist in NAU_T_1),
# "right_only" (existed in NAU_T_1 but not anymore), "both" (exists in both dataframes)

merged = pd.merge(between_4,NAU_T_1, on=["ISIN"], how="outer", indicator=True)

#get the month-end-day of the calc date (important when we run the prov. NAU some days in advance)


calc_date_pd = pd.to_datetime(str(calc_date), format='%Y%m%d')

input_dt = calc_date_pd
next_month = input_dt.replace(day=28) + timedelta(days=4)
calc_date_pd = next_month - timedelta(days=next_month.day)
calc_date_pd = np.datetime64(calc_date_pd)

#if "both" but not "Grenzwertig" -> take it and create "Grenzwertig" with the current calc date

mask = (merged["_merge"] == "both") & (merged["Auf der Kippe?"] != "Grenzwertig")
new_in = merged.loc[mask].copy()

new_in["Auf der Kippe?"] = "Grenzwertig"
new_in["Seit wann"] = calc_date_pd
new_in = new_in.rename(columns={"MSCI ID_x":"MSCI ID","Name_x":"Name", "Score_x":"Score", "Rating_x":"Rating"})
new_in["Seit wann"] = pd.to_datetime(new_in["Seit wann"])

top_45.loc[:, "Auf der Kippe?"] = np.nan
top_45.loc[:, "Seit wann"] = np.nan


NAU = pd.concat([top_45, new_in], join="inner")
#NAU[~NAU["Seit wann"].isnull()]

#if "both" and "Grenzwertig" -> take it with as "Grenzwertig" with the existing date

in_already = merged[merged["_merge"] == "both"][merged["Auf der Kippe?"] == "Grenzwertig"]
in_already = in_already.rename(columns={"MSCI ID_x":"MSCI ID","Name_x":"Name", "Score_x":"Score", "Rating_x":"Rating"})
in_already = in_already[["Score", "Rating", "MSCI ID", "Name", "ISIN", "Auf der Kippe?", "Seit wann", "GICS_SUB_IND", "msci_security_code"]]

NAU_new = pd.concat([NAU, in_already], join="inner")

#in_already = in_already[["ISSUER_NAME", "ISIN", ]]

#%% # delete those that are now for over 6 months stated as Grenzwertig

from pandas.tseries.offsets import DateOffset

date_check = NAU_new.copy()
date_check["Seit wann"] = date_check["Seit wann"] + MonthEnd(0)
date_check["Seit wann"] = date_check["Seit wann"] + DateOffset(months=6)
date_check = date_check[date_check["Seit wann"] == calc_date_pd]

NAU_pre_final = pd.concat([NAU_new, date_check], join="inner")
NAU_pre_final = NAU_pre_final.drop_duplicates(subset=["ISIN"], keep=False)

#NAU_pre_final.to_excel("NAU_pre_final.xlsx")

#%% creating the out (constituents not in NAU anymore) and in (constituents new in NAU) lists

differences = pd.merge(NAU_pre_final, NAU_T_1, how="outer", on="ISIN", indicator=True)
outs = differences[differences["_merge"] == "right_only"]
ins = differences[differences["_merge"] == "left_only"]

ins = ins.rename(columns={"Name_x": "Name", "MSCI ID_x": "MSCI ID", "Auf der Kippe?_x": "Auf der Kippe?", "Score_x":"Score", "Rating_x":"Rating", "Seit wann_x":"Seit wann"})
ins = ins[["Name", "MSCI ID","Score", "Auf der Kippe?", "Rating", "Seit wann","ISIN", "msci_security_code"]]

outs = outs.rename(columns={"Name_y": "Name", "MSCI ID_y": "MSCI ID", "Auf der Kippe?_y": "Auf der Kippe?", "Score_x":"Score", "Rating_y":"Rating", "Seit wann_y":"Seit wann"})
outs = outs[["Name", "MSCI ID","Score", "Auf der Kippe?", "Rating", "Seit wann", "ISIN", "msci_security_code"]]

#%% updating outs with current ratings and scores
index_data_esg1 = index_data_esg.rename(columns={"ISSUER_ISIN": "ISIN"})
outs_merged = pd.merge(outs, index_data_esg1, how="inner", on="ISIN")
outs_merged = outs_merged[["Name", "MSCI ID","ISIN", "Auf der Kippe?", "Seit wann", "INDUSTRY_ADJUSTED_SCORE","IVA_COMPANY_RATING", "msci_security_code_y"]]
outs_merged = outs_merged.rename(columns={"INDUSTRY_ADJUSTED_SCORE": "Score", "IVA_COMPANY_RATING": "Rating", "msci_security_code_y":"msci_security_code"})

outs = pd.merge(outs, outs_merged, how="outer", on="ISIN")
outs = outs.rename(columns={"Name_x":"Name", "MSCI ID_x":"MSCI ID", "Score_y":"Score", "Auf der Kippe?_x":"Auf der Kippe?", "Rating_y":"Rating", "Seit wann_x":"Seit wann", "msci_security_code_x":"msci_security_code"})
outs = outs[["Name", "MSCI ID","Score", "Auf der Kippe?", "Rating", "Seit wann", "ISIN", "msci_security_code"]]

# Getting Region and GICS Sector from NAU_T_1
outs = pd.merge(outs, NAU_T_1, how="inner", on="ISIN")
outs = outs.rename(columns={"Name_x":"Name", "MSCI ID_x":"MSCI ID", "Score_x":"Score", "Auf der Kippe?_x":"Auf der Kippe?", "Rating_x":"Rating", "Seit wann_x":"Seit wann"})

#%% showing constituents that are in 150er list (research coverage) that would fall out of the NAU

Aktienliste_150 = pd.read_excel(r"R:\Spezial\Bloomberg_IC\Bloomberg_IC\Aktienliste_150.xlsm", sheet_name='alle', usecols=["BLKB Research Universum","Unnamed: 3"])

Aktienliste_150 = Aktienliste_150.iloc[2:,:].rename(columns={"BLKB Research Universum": "ISIN", "Unnamed: 3": "Name"}).reset_index(drop=True)

outs_150er = pd.merge(outs, Aktienliste_150, how="inner", on="ISIN")

#%%
## create a list with those that are in 150er list and grenzwertig, as well as those that are overruled and under review

NAU_grenzwertig = NAU_pre_final[NAU_pre_final["Auf der Kippe?"]=="Grenzwertig"]
grenzwertig_150er = pd.merge(NAU_grenzwertig, Aktienliste_150, how="inner", on="ISIN").rename(columns={"Name_x":"Name"})
grenzwertig_150er.drop(columns={"GICS_SUB_IND", "Name_y"}, inplace=True)

overrulled_grenzwertig_150er = pd.concat([grenzwertig_150er, overruled], ignore_index=True)

condition1 = overrulled_grenzwertig_150er["Rating"] == "OVERRULED"
condition2 = overrulled_grenzwertig_150er["Rating"] == "REVIEW"
condition3 = overrulled_grenzwertig_150er["Auf der Kippe?"] == "Grenzwertig"

overrulled_grenzwertig_150er["Läuft ab"] = pd.NaT

overrulled_grenzwertig_150er.loc[condition1, "Läuft ab"] = overrulled_grenzwertig_150er.loc[condition1,"Seit wann"] + pd.DateOffset(months=12) + MonthEnd(0)
overrulled_grenzwertig_150er.loc[condition2, "Läuft ab"] = overrulled_grenzwertig_150er.loc[condition2,"Seit wann"] + pd.DateOffset(months=2) + MonthEnd(0)
overrulled_grenzwertig_150er.loc[condition3, "Läuft ab"] = overrulled_grenzwertig_150er.loc[condition3,"Seit wann"] + pd.DateOffset(months=6) + MonthEnd(0)

df_msci = df_index[["msci_security_code","isin"]].rename(columns={"isin":"ISIN"})
overrulled_grenzwertig_150er.drop(columns={"msci_security_code"}, inplace=True)
overrulled_grenzwertig_150er = pd.merge(overrulled_grenzwertig_150er, df_msci, on="ISIN", how="inner")
overruled = pd.merge(overruled, df_msci, on="ISIN", how="left")

#%%
## Getting GICS sectors for each constituent

#MSCI ACI IMI Constituents
df_cons = df_index[['msci_security_code']]

#get gics code - gics name mapping from same constituents endpoint but different dataset

url_index1 = "https://api.msci.com/index/constituents/v1.0/indexes/" + str(index_code) + "/initial?calc_date=" + str(
    calc_date) + "&as_of_date=" + str(as_of_date) + \
             "&output=SECTOR_WEIGHT"

headers = {'content-type': "application/x-www-form-urlencoded",
           'accept': "application/json", 'accept-encoding': "deflate,gzip",
           "Authorization": "Basic %s" % b64login_index}

response = requests.request("GET", url_index1, headers=headers, verify=path_zert)
data_index = response.json()

df_gics_map = data_index['indexConstituents'][0]['SECTOR_WEIGHT']
df_gics_map = pd.json_normalize(df_gics_map)
df_gics_map['sector'] = df_gics_map['sector'].astype('str')
df_gics_map=df_gics_map[['sector','sector_name']]
df_gics_map.rename(columns={'sector': 'GICS.sector', 'sector_name': 'GICS.sector_name'}, inplace=True)

# functions to get index form index API

headers = {'content-type': "application/x-www-form-urlencoded",
           'accept': "application/json", 'accept-encoding': "deflate,gzip",
           "Authorization": "Basic %s" % b64login_index}

params = {'calc_date': calc_date, 'as_of_date': as_of_date,'output':'GICS','distribution_zone':'WORLD','cumulative':'true'}
r = requests.get('https://api.msci.com/index/secmaster/security/v2.0/securities/ALL/marketopen', headers=headers, params=params, verify=path_zert)
df_secs=pd.DataFrame.from_dict(pd.json_normalize(r.json()['securities']), orient='columns')
df_secs=df_secs[['msci_security_code','GICS.sector']]
df_secs['GICS.sector'] = df_secs['GICS.sector'].astype('str')

#join the dataframes together

df_cons_with_gics = pd.merge(df_cons, df_secs, how='left', on=['msci_security_code'])
df_cons_with_gics_and_gics_name = pd.merge(df_cons_with_gics, df_gics_map, how='left', on=['GICS.sector'])
df_cons_with_gics_and_gics_name.drop(columns={"GICS.sector"}, inplace=True)
df_cons_with_gics_and_gics_name.rename(columns={"GICS.sector_name": "Sector"}, inplace=True)

# add GICS sectors to the other datasets
NAU_pre_final = pd.merge(NAU_pre_final, df_cons_with_gics_and_gics_name, how="inner", on="msci_security_code")
ins = pd.merge(ins, df_cons_with_gics_and_gics_name, how="inner", on="msci_security_code")
overrulled_grenzwertig_150er = pd.merge(overrulled_grenzwertig_150er, df_cons_with_gics_and_gics_name, how="inner", on="msci_security_code")
#overruled = pd.merge(overruled, df_cons_with_gics_and_gics_name, how="left", on="msci_security_code")


#%% Mapping of regions (Nordamerika, Europa, Schweiz, World) to the dataframes

regionen_mapping = pd.read_excel(r'Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Aktien\Mapping Regionen.xlsx')
df_region = df_index[['ISO_country_symbol', "isin"]]
df_region = pd.merge(df_region, regionen_mapping, on="ISO_country_symbol", how="inner")
df_region = df_region.drop(columns={"ISO_country_symbol"}).rename(columns={"isin":"ISIN"})

NAU_pre_final = pd.merge(NAU_pre_final, df_region, on="ISIN", how="inner")
ins = pd.merge(ins, df_region, on="ISIN", how="inner")
overrulled_grenzwertig_150er = pd.merge(overrulled_grenzwertig_150er, df_region, on="ISIN", how="inner")
overrulled_grenzwertig_150er= overrulled_grenzwertig_150er.rename(columns={"Region_y":"Region", "Sector_y":"Sector"}).drop(columns={"Region_x", "Sector_x"})

#%%
# add overruled/reviewed ones to the end of the dataframe
overruled.loc[overruled["ISIN"]=="DE000A3H23N4", "Sector" ] = NAU_pre_final.loc[NAU_pre_final["ISIN"]=="DE0008404005", "Sector" ].values[0]
NAU_pre_final = pd.concat([NAU_pre_final, overruled], ignore_index=True)
NAU_pre_final["Seit wann"] = pd.to_datetime(NAU_pre_final["Seit wann"])

# indicate those that are in the 150er list (research coverage)
NAU_pre_final1 = pd.merge(NAU_pre_final, Aktienliste_150, how="outer", on="ISIN", indicator=True)
NAU_pre_final1['_merge'] = NAU_pre_final1['_merge'].astype(str)
NAU_pre_final1.loc[NAU_pre_final1["_merge"]=="both", "_merge"] = "Ja"
NAU_pre_final1.loc[NAU_pre_final1["_merge"]=="left_only", "_merge"] = np.nan
NAU_pre_final1.loc[NAU_pre_final1["_merge"]=="right_only", "_merge"] = np.nan
NAU_pre_final1= NAU_pre_final1.drop(columns={"Name_y"}).rename(columns={"Name_x":"Name", "_merge":"150er Liste"})

# cleaning dataframes for export

NAU_pre_final1.drop(columns={"GICS_SUB_IND", "msci_security_code"}, inplace=True)
ins.drop(columns={"msci_security_code"}, inplace=True)
outs.drop(columns={"msci_security_code"}, inplace=True)

NAU_pre_final1["Seit wann"] = NAU_pre_final1["Seit wann"].dt.date
outs["Seit wann"] = outs["Seit wann"].dt.date
overrulled_grenzwertig_150er["Seit wann"] = overrulled_grenzwertig_150er["Seit wann"].dt.date
overrulled_grenzwertig_150er["Läuft ab"] = overrulled_grenzwertig_150er["Läuft ab"].dt.date
NAU_T_1["Seit wann"] = NAU_T_1["Seit wann"].dt.date



NAU_final = NAU_pre_final1.loc[:,["Score", "Rating", "MSCI ID", "Name", "ISIN", "Region", "Sector", "Auf der Kippe?", "Seit wann", "150er Liste"]]
ins = ins.loc[:,["Score", "Rating", "MSCI ID", "Name", "ISIN", "Region", "Sector", "Auf der Kippe?", "Seit wann"]]
outs = outs.loc[:,["Score", "Rating", "MSCI ID", "Name", "ISIN", "Region", "Sector", "Auf der Kippe?", "Seit wann"]]
overrulled_grenzwertig_150er = overrulled_grenzwertig_150er.loc[:,["Score", "Rating", "MSCI ID", "Name", "ISIN", "Region", "Sector", "Auf der Kippe?", "Seit wann", "Läuft ab"]]

NAU_final = NAU_final.dropna(subset=['MSCI ID'])
#%%

top_45.loc[top_45.index[0], "ISIN"] = "AT0000A18XM4"
top_45.loc[top_45.index[1], "ISIN"] = "DE000A3H23N4"
NAU_final.loc[NAU_final.index[0], "ISIN"] = "AT0000A18XM4"
NAU_final.loc[NAU_final.index[1], "ISIN"] = "DE000A3H23N4"

#%%
# check if those under overruled or review appear now in top_45
# if one (or more) appears in top_45 (i.e., is listed twice), remove it (last matching one in NAU_final) and keep the one appearing in top_45

check_overruled = overruled.merge(top_45, how="inner", on="ISIN")
overruled_ISINs = set(check_overruled["ISIN"])
last_matching_indices = []

for ISIN in overruled_ISINs:
    matching_indices = NAU_final[NAU_final["ISIN"] == ISIN].index
    if len(matching_indices) > 0:
        last_matching_indices.append(matching_indices[-1])

last_matching_indices = NAU_final.index.isin(last_matching_indices)
NAU_final.drop(last_matching_indices.nonzero()[0], inplace=True, errors="ignore")

# also delete it from the overrulled_grenzwertig_150er
overrulled_grenzwertig_150er = overrulled_grenzwertig_150er[~overrulled_grenzwertig_150er["ISIN"].isin(check_overruled["ISIN"])]


#%%
# export: creating for each of the following dataframes and excel-sheet within the final excel file
with pd.ExcelWriter("NAU_final6_4.xlsx") as writer:

    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    NAU_final.to_excel(writer, sheet_name="BoU", index=False)
    ins.to_excel(writer, sheet_name="In", index=False)
    outs.to_excel(writer, sheet_name="Out", index=False)
    NAU_T_1.to_excel(writer, sheet_name="T-1", index=False)
    overrulled_grenzwertig_150er.to_excel(writer, sheet_name="Overruled+Grenzwertig 150er", index=False)

#%%
# showing constituents that are in 150er list (research coverage) that would fall out of the NAU

print(outs_150er[["Name_x", "ISIN"]])

#%%
# yaml factor list
with open("Adjustments Aktien-NAU.yml", 'r') as stream:
    adjustments = yaml.safe_load(stream)

new_reviewed = adjustments['new review']
new_overruled = adjustments['new overruling']
new_deletion = adjustments['new deletion']



#%%

condition1 = NAU_final["ISIN"].isin(new_reviewed)

NAU_final.loc[condition1, ["Score","Rating"]] = "REVIEW"

NAU_final.loc[condition1, "Seit wann"] = calc_date_pd
NAU_final["Seit wann"] = pd.to_datetime(NAU_final["Seit wann"])
NAU_final["Seit wann"] = NAU_final["Seit wann"] + MonthEnd(0)
NAU_final.loc[condition1, "Seit wann"] = NAU_final.loc[condition2, "Seit wann"] .dt.date

#%%

condition2 = NAU_final["ISIN"].isin(new_overruled)

NAU_final.loc[condition2, ["Score","Rating"]] = "OVERRULED"

NAU_final.loc[condition2, "Seit wann"] = calc_date_pd
NAU_final["Seit wann"] = pd.to_datetime(NAU_final["Seit wann"])
NAU_final["Seit wann"] = NAU_final["Seit wann"] + MonthEnd(0)
NAU_final.loc[condition2, "Seit wann"] = NAU_final.loc[condition2, "Seit wann"] .dt.date

