import gspread   #gpsread is a Python client library for the Google Sheets API
from oauth2client.service_account import ServiceAccountCredentials # OAuth is an open standard for token-based authentication and authorization on the Internet
import pprint

#authorize the client object, which takes credentials, which are a combination of Credentials in a JSON file and scope of access (google sheets and the google drive API)
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('Python-a44fefaf4a10.json', scope)
client = gspread.authorize(creds)

sheet = client.open('python_example').sheet1

#data = sheet.range('C37:E61')
#print(data)

price_targets, probs, pe_vols = [], [], []
#get price_targets
data = sheet.range('C37:C61')
for cell in data:
    price_targets.append(cell.value)

#get probabilities
data = sheet.range('D37:D61')
for cell in data:
    probs.append(cell.value)

#get post_event vols
data = sheet.range('E37:E61')
for cell in data:
    pe_vols.append(cell.value)

#create dictionary to hold price_targets, probabilities, post_event vols
event_dict = {}
event_dict['price_targets'] = price_targets
event_dict['probs'] = probs
event_dict['pe_vols'] = pe_vols

print(event_dict)

#insert qualitative information into the event_dict
event_dict['stock']                 = sheet.cell(1,2).value
event_dict['event_name']            = sheet.cell(2,2).value
event_dict['event_timing']          = sheet.cell(3,2).value
event_dict['model_type']            = sheet.cell(4,2).value
event_dict['reference_price']       = sheet.cell(5,2).value
event_dict['ne_volatility']         = sheet.cell(6,2).value
event_dict['expiry']                = sheet.cell(7,2).value
event_dict['prob_event_by_expiry']  = sheet.cell(8,2).value

print(event_dict)

#print(price_targets, '\n', probs, '\n', pe_vols)

#pprint.pprint(data)
#print(type(data), type(data[0]))


#reqs = {'requests': [
#    "range": {
#        "sheetID": 0,
#        "startColumnIndex": 0
#        "endColumnIndex": 5,
#        "startRowIndex":37,
#        "endRowIndex": 62.
#    }
#]}
#
#SHEETS.spreadsheets().batchupdate(
#        spreadsheetID=SHEET_ID, body=reqs).execute()
