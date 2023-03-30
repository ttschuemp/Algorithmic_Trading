from base64 import b64encode
import requests
import pandas as pd
import json
import config_msci_api as config
import yaml
import numpy as np
from datetime import datetime, timedelta

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


df = pd.DataFrame.from_dict({"reference": ["issuerid", "issuer_id", "issuername", "issuer_name", "level"]})
reference = df["reference"].to_list()


# function to get ESG data from ESG API

url = "https://api.msci.com/esg/data/v1.0/issuers"
headers = { "Authorization": "Basic %s" % b64login,
            'content-type': "application/json" }

# ESG factor list BLKB exclusion criteria
factor_list_esg_exclusions = factor_list['exclusion criteria']

payload = json.dumps({"factor_name_list": factor_list_esg_exclusions,
                      "limit" : 10000,
                      "coverage": "esg_ratings",
                      "parent_child":"full_parent_child",
                      "reference_column_list": reference
                      })

response = requests.request("POST", url, data=payload, headers=headers, verify=path_zert)
data = response.json()

results = data['result']['issuers']
df_result = pd.DataFrame.from_dict(pd.json_normalize(results), orient='columns')

keyorder = factor_list_esg_exclusions
# important to know: with reindexing columns from keyorder, you only take those columns out that are specified in the yml-data "factor_list_esg_exclusion"
df_data_esg1 = df_result.reindex(columns=keyorder)

payload = json.dumps({"factor_name_list": factor_list_esg_exclusions,
                      "limit" : 10000,
                      "offset": 10000,
                      "coverage": "esg_ratings",
                      "parent_child": "full_parent_child",
                      "reference_column_list": reference
                      })

response = requests.request("POST", url, data=payload, headers=headers, verify=path_zert)
data = response.json()
results = data['result']['issuers']
df_result = pd.DataFrame.from_dict(pd.json_normalize(results), orient='columns')

keyorder = factor_list_esg_exclusions
df_data_esg2 = df_result.reindex(columns=keyorder)

#getting governments, but only those that are under "allowed countries"

countries = factor_list['allowed countries']


payload = json.dumps({"factor_name_list": factor_list_esg_exclusions,
                      "limit" : 10000,
                      "coverage": "government_ratings",
                      "parent_child":"full_parent_child",
                      "reference_column_list": reference,
                      "country_code_list": countries
                      })

response = requests.request("POST", url, data=payload, headers=headers, verify=path_zert)
data = response.json()

results = data['result']['issuers']
df_result = pd.DataFrame.from_dict(pd.json_normalize(results), orient='columns')

keyorder = factor_list_esg_exclusions
# important to know: with reindexing columns from keyorder, you only take those columns out that are specified in the yml-data "factor_list_esg_exclusion"
df_data_esg3 = df_result.reindex(columns=keyorder)


df_data_esg = pd.concat([df_data_esg1, df_data_esg2, df_data_esg3], ignore_index=True)

# export ISINs for SIX iD to get GK-numbers
df_isins = df_data_esg[["ISSUER_ISIN"]].dropna()

df_isins.to_csv("R:/Spezial/Bloomberg_IC/Bloomberg_IC/Sascha Gut/FixedIncome/Bondlisten/TelekursImportExport/df_isins.csv", index=False, header=False)

#%%

df_filtered = df_data_esg[df_data_esg["IVA_COMPANY_RATING"].str.contains("AAA|AA|A|BBB", na=False) | df_data_esg["GOVERNMENT_ESG_RATING"].str.contains("AAA|AA|A|BBB", na=False)]

#%%

gics_sub_exclusion2 = ["Airlines", "Aerospace & Defense ", "Electric Utilities"]
data_filtered = df_filtered[~df_filtered['GICS_SUB_IND'].isin(gics_sub_exclusion2)]

# fill nan weapon revenue percentage with 0
weapon_exclusion = ['GAM_OPER_MAX_REV_PCT', 'WEAP_MAX_REV_PCT',
                    'FIREARM_PROD_MAX_REV_PCT', 'FIREARM_RET_MAX_REV_PCT']

data_filtered[weapon_exclusion] = data_filtered[weapon_exclusion].fillna(0)

# exclusion of weapon producer with revenue share > 5%
data_filtered = data_filtered[(data_filtered['GAM_OPER_MAX_REV_PCT'] < 5) & (data_filtered['WEAP_MAX_REV_PCT'] < 5)
                              & (data_filtered['FIREARM_PROD_MAX_REV_PCT'] < 5) & (data_filtered['FIREARM_RET_MAX_REV_PCT'] < 5)]

# BLKB exclusion criteria
esg_exclusion_criterias = ['TOB_PRODUCER','NUC_UTILITY','NUC_URANIUM_MINE','INDUSTRY_ENERGY_PRODUCER',
                           'INDUSTRY_ENERGY_APPLICATION','COAL_RESERVES','GMO_AGRICULTURE','WEAP_GPRODUCER',
                           'AE_PRODUCER','CB_MANUFACTURER', 'LM_MANUFACTURER', 'WEAP_BIO_CHEM_SYSTEM', 'DU_WEAP',
                           'WEAP_NUC_SYSTEM']

# government exclusion criteria
# fill None with 0 and exclude if value >= 5 (i.e., include all below 5)
data_filtered['GOVERNMENT_RAW_CIVIL_LIBERT'] = data_filtered['GOVERNMENT_RAW_CIVIL_LIBERT'].fillna(0)
data_filtered = data_filtered[(data_filtered['GOVERNMENT_RAW_CIVIL_LIBERT'] < 5)]

# change None to False
data_filtered.fillna(False, inplace = True)

# remove all rows containing exlusion criterias
data_filtered = data_filtered[~data_filtered[esg_exclusion_criterias].isin([True]).any(axis=1)]
data_filtered = data_filtered[~data_filtered['GOVERNMENT_POLIT_RIGHTS'].isin(["Yes"])]

#drop duplicates
data_filtered.drop_duplicates(inplace=True)

#%%

# get the latest (last saved) Aktien-NAU file from the folder
from pathlib import Path
import os

paths = [(p.stat().st_mtime, p) for p in Path("Q:/PBIS_IC_alle_MA/Nachhaltigkeit (SRI)/Universum Aktien/BLKB NAU Aktien").iterdir() if p.suffix == ".xlsx"]
paths = sorted(paths, key=lambda x: x[0], reverse=True)
last = paths[0][1].name
last = str(last)
last = last[2:]

path = "Q:/PBIS_IC_alle_MA/Nachhaltigkeit (SRI)/Universum Aktien/BLKB NAU Aktien/"
fullpath = os.path.join(path, last)
aktien_nau = pd.read_excel(fullpath, sheet_name="BoU")

msci_acwi = pd.read_excel(r"Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Aktien\NAUdefinitivabAugust2022 - NEU.xlsm", sheet_name='Input MSCI Index Monitor')

aktien_nau.rename(columns={"MSCI ID":"ISSUERID"}, inplace=True)

#%%
# check if issuer is in MSCI ACWI -> if not, keep it. If yes, check if it is in the current Aktien_NAU -> if not, kick it out of the list
#here check those that are "right_only" -> why?
in_acwi = pd.merge(data_filtered, msci_acwi, on="ISSUERID", how="inner")
in_nau = pd.merge(in_acwi, aktien_nau, on="ISSUERID", how="outer", indicator=True)
out_nau = in_nau[in_nau["_merge"]=="left_only"]

out = data_filtered["ISSUERID"].isin(out_nau["ISSUERID"])

nau_filtered = data_filtered[~out]


#%%

nau_filtered.loc[nau_filtered["ISSUER_ISIN"]==False, "ISSUER_ISIN"] = np.nan

obli_nau = nau_filtered.dropna(subset=["ISSUER_ISIN"])
obli_nau.to_excel("obli_nau3.xlsx", index=False)

#%%

# export ISINs to get GKs from SIX iD and save it as GK Liste under the following folder: Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Obligationen\

obli_nau["ISSUER_ISIN"].to_csv("R:\Spezial\Bloomberg_IC\Bloomberg_IC\Sascha Gut\FixedIncome\Bondlisten\TelekursImportExport\BOND_ISINs.csv",  index=False, header=False)

#%%
# read the excel file with gk's from SIX iD

obli_nau_gk = pd.read_excel(r"Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Obligationen\GK Liste.xlsx")

obli_nau_gk = obli_nau_gk[["ISIN", "U-GKey"]]
obli_nau_gk= obli_nau_gk.rename(columns={"ISIN":"ISSUER_ISIN", "U-GKey":"GK"})

obli_nau_gk = obli_nau_gk.dropna(subset=["GK"])
obli_nau_gk["GK"] = obli_nau_gk["GK"].astype("int32")

obli_nau_gk = pd.merge(obli_nau, obli_nau_gk, on="ISSUER_ISIN", how="inner")


#%%
# check if we have bonds from countries or regions, that are trading at SIX, but are not included in the universe (before exclusions)

# loading all bonds that are trading via SIX (Anleihen-Explorer) into the data.
# source: https://www.six-group.com/de/products-services/the-swiss-stock-exchange/market-data/bonds/bond-explorer.html

six_bonds = pd.read_excel(r"Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Obligationen\SIX Bonds.xlsx")
six_bonds = six_bonds[["ISIN", "IndustrySectorDesc"]]
df = six_bonds["ISIN"].to_list()

url = "https://api.msci.com/esg/data/v1.0/issuers"
headers = { "Authorization": "Basic %s" % b64login,
            'content-type': "application/json" }

# ESG factor list BLKB exclusion criteria
factor_list_esg_exclusions = factor_list['exclusion criteria']

payload = json.dumps({"factor_name_list": factor_list_esg_exclusions,
                      "limit" : 10000,
                      "issuer_identifier_list": df,
                      "parent_child":"full_parent_child",
                      "reference_column_list": reference
                      })

response = requests.request("POST", url, data=payload, headers=headers, verify=path_zert)
data = response.json()

results = data['result']['issuers']
df_result = pd.DataFrame.from_dict(pd.json_normalize(results), orient='columns')

keyorder = factor_list_esg_exclusions
df_six_data = df_result.reindex(columns=keyorder)

df_six_data = df_six_data.dropna(subset=["CLIENT_IDENTIFIER"])
df_six_data = df_six_data.drop_duplicates(subset=["ISSUERID"])

df_merged = pd.merge(df_six_data, df_data_esg, on="ISSUERID", how="outer", indicator=True)

df_merged = df_merged[df_merged["_merge"]=="left_only"]
countries_in = df_merged["ISSUER_CNTRY_DOMICILE_x"].isin(countries)
df_merged = df_merged[countries_in]

# adding own mapping to the data where mostly Swiss Canton's are listed, for which we take the rating of the Schweizer Eidgenossenschaft
mapping = pd.read_excel(r"Q:\PBIS_IC_alle_MA\IC-Team\iQ\iQ\BLKB_NAU\Nachhaltigkeitsuniversum Obligationen\New Mapping.xlsx")

df_merged2 = pd.merge(df_merged, mapping, on="ISSUERID", how="outer", indicator="Existent")

df_merged2 = df_merged2[df_merged2["Existent"]=="left_only"]
df_merged2.fillna(False, inplace = True)

six_bonds_govs = six_bonds[(six_bonds["IndustrySectorDesc"]=="Regions, Cantons, Provinces, etc.") | (six_bonds["IndustrySectorDesc"]=="Countries")]
six_bonds_govs = six_bonds_govs.rename(columns={"ISIN":"CLIENT_IDENTIFIER_x"})

six_check = pd.merge(df_merged2, six_bonds_govs, on="CLIENT_IDENTIFIER_x", how="inner")

# if new issuers appear -> add them to mapping
six_check

#%%
# add mapped issuers to the obli_nau

obli_nau_mapped = pd.concat([obli_nau_gk, mapping],ignore_index=True)

# create one column "Rating", incorporating both company and gov ratings
obli_nau_mapped["IVA_COMPANY_RATING"].fillna(0, inplace=True)
obli_nau_mapped.loc[obli_nau_mapped["IVA_COMPANY_RATING"].isnull(), "Rating"] = obli_nau_mapped["GOVERNMENT_ESG_RATING"]
obli_nau_mapped.loc[obli_nau_mapped["IVA_COMPANY_RATING"]==False, "Rating"] = obli_nau_mapped["GOVERNMENT_ESG_RATING"]
obli_nau_mapped.loc[obli_nau_mapped["IVA_COMPANY_RATING"]!=False, "Rating"] = obli_nau_mapped["IVA_COMPANY_RATING"]


#%%
# get the latest (last saved) Bond-NAU file from the folder

paths = [(p.stat().st_mtime, p) for p in Path("Q:/PBIS_IC_alle_MA/Nachhaltigkeit (SRI)/Universum Obligationen/BLKB NAU Obligationen").iterdir() if p.suffix == ".xlsx"]
paths = sorted(paths, key=lambda x: x[0], reverse=True)
last = paths[0][1].name
last = str(last)
last = last[2:]

path = "Q:/PBIS_IC_alle_MA/Nachhaltigkeit (SRI)/Universum Obligationen/BLKB NAU Obligationen/"
fullpath = os.path.join(path, last)
bond_nau_t_1 = pd.read_excel(fullpath, sheet_name="T-1")
bond_nau_t_1["GK"] = bond_nau_t_1["GK"].astype("int32")

#%%

differences = pd.merge(obli_nau_mapped, bond_nau_t_1, how="outer", on="GK", indicator=True)
outs = differences[differences["_merge"] == "right_only"]
ins = differences[differences["_merge"] == "left_only"]


#%%
# updating outs with current ratings and scores
outs = outs.drop(columns={"ISSUER_ISIN"})
outs = outs.rename(columns={"ISIN":"ISSUER_ISIN"})
outs = pd.merge(outs, df_data_esg, how="inner", on="ISSUER_ISIN")

# cleaning data for excel export
outs = outs.rename(columns={"Rating_y": "Rating", "ISSUER_ISIN":"ISIN"})
outs = outs[["ISIN", "GK", "Name", "Rating", "Country"]]
ins = ins.drop(columns={"Country", "ISIN", "Name"})
ins = ins.rename(columns={"ISSUER_NAME": "Name", "ISSUER_ISIN":"ISIN", "ISSUER_CNTRY_DOMICILE":"Country", "Rating_x":"Rating"})
ins = ins[["ISIN", "GK", "Name", "Rating", "Country"]]

obli_nau_mapped_clean = obli_nau_mapped.rename(columns={"ISSUER_ISIN":"ISIN", "ISSUER_NAME":"Name", "ISSUER_CNTRY_DOMICILE":"Country"})

obli_nau_mapped_clean = obli_nau_mapped_clean[["ISIN", "GK", "Name", "Rating", "Country"]]
obli_nau_mapped_clean["GK"] = obli_nau_mapped_clean.GK.astype("int32")

obli_nau_mapped_clean.drop_duplicates(subset=["GK"], inplace=True)


#%%

# creating for each of the following dataframes and excel-sheet within the final excel file
with pd.ExcelWriter("bond_nau.xlsx") as writer:

    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    obli_nau_mapped_clean.to_excel(writer, sheet_name="Obli NAU", index=False)
    ins.to_excel(writer, sheet_name="In", index=False)
    outs.to_excel(writer, sheet_name="Out", index=False)
    bond_nau_t_1.to_excel(writer, sheet_name="T-1", index=False)
#%%
