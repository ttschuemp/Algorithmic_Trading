# SIX API

myUser = 'blkbapid03'
myPw = 'Blkbapid03'
myCustId = 'CH111'

# get credentials
import requests
import xmltodict
import xmlschema
import nested_lookup as nl

# schema
# xs = xmlschema.XMLSchema('https://apidintegra.tkfweb.com/apid/xml/xrf.xsd')

# myLink = f'https://apidintegra.tkfweb.com/apid/request?method=login&ci=apiD&ui={myCustId}-{myUser}&pwd={myPw}'

myLink = f'https://apidprod.tkfweb.com/apid/request?method=login&ci=apiD&ui={myCustId}-{myUser}&pwd={myPw}'

myResponse = requests.get(myLink, verify="C:\DevLab\zscaler_root.cer")

myResponseDict = xmltodict.parse(myResponse.text)
mySessionID = myResponseDict['XRF']['A']['@v']
print(f'SessionID = {mySessionID}')

#%%
# get data
# mySearchKey = 'method=getXMLTable&name=701&lang=1'
# mySearchKey = 'method=getXMLTable&name=700&lang=1'
# mySearchKey = 'method=getXMLTable&name=713&lang=1'

# mySearchKey = 'method=getTimeSeries&ik=1222171,380,1&pk=1&mode=tick&max=3& ' \
#               'time_from=15:00&time_to=16:00&date_from=13.06.2011&date_to=14.06.2011&id=[session-key]&ci=' \
#               '[client-id]&ui=[user-id]https://[server-url]/request?method=getTimeSeries&ik=1222171,380,1&pk=' \
#               '1&mode=tick&max=3& time_from=15:00&time_to=16:00&date_from=13.06.2021&date_to=14.06.2021'

mySearchKey = 'method=getListingListData&pk=516'
# mySearchKey = 'method=getListingData&ik1=3936085,625,333&pk=33,581,2'
# listing class
# mySearchKey = 'method=getListingData&ik1=110435293,880,1&mk=5'

# mySearchKey = 'method=getHikuData&ik=998089,4,1&pk=3,1&date_from=01.01.2022&date_to=15.01.2022'

# mySearchKey = 'method=getHikuData&ik=950498,393,1&pk=3,1&date_from=01.01.2022&date_to=31.03.2022'

# mySearchKey = 'method=getHikuData&ik1=998089,4,1&ik2=277340,393,1&pk=3,1&date_from=13.07.2022&date_to=13.07.2022'

# mySearchKey = 'method=getHikuData&ik=1222171,380,1&pts=1&pk= 2,1;3,1;12,0;14,0&date_from=20.12.2013&date_to=10.01.2014'

# mySearchKey = 'method=getHikuData&ik=42980817,880,1&pk=avail&date_from=01.01.2022&date_to=31.01.2022'

# mySearchKey = 'method=getHikuData&ik=42980817,880,1&pk=3,1&date_from=01.01.2022&date_to=31.01.2022'

# mySearchKey = 'method=getHikuData&ik=50659597,880,333&pk=avail&date_from=28.09.2022&date_to=28.09.2022'

# mySearchKey = 'method=getHikuData&ik=32976392,190,1&pk=avail&date_from=26.10.2022&date_to=26.10.2022'
# mySearchKey = 'method=getHikuData&ik=32976392,393,1&pk=avail&date_from=06.12.2022&date_to=06.12.2022'
# mySearchKey = 'method=getHikuData&ik=39592984,4411,1&pk=avail&date_from=03.01.2022&date_to=03.01.2022'

# myLink = f'https://apidintegra.tkfweb.com/apid/request?{mySearchKey}&id={mySessionID}&ui={myCustId}-{myUser}'

myLink = f'https://apidprod.tkfweb.com/apid/request?{mySearchKey}&id={mySessionID}&ui={myCustId}-{myUser}'

myResponse = requests.get(myLink, verify="C:\DevLab\zscaler_root.cer")

myResponseDict = xmltodict.parse(myResponse.text)

# myLastSMI = myResponseDict['XRF']['IL']['I']['HL']['HD']['P']['@v']

myLastSMI2 = nl.nested_lookup('@v',myResponseDict)
myLastDates = nl.nested_lookup('@d',myResponseDict)

f = open(r'Q:\PBIS_IC_alle_MA\Investment, Performance & Risk Controlling\Diverses\Deployment\test_six_michi\myoutput.txt', 'a')
f.write(str(myLastDates))
f.close()
print('test')