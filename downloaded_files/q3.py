
from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io




from flask import Flask, request
from flasgger import Swagger
import pickle
#import sklearn
#import scikit_posthocs
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, Markup




# Loading the crop Recommendation Model
crop_recommendation_model_path = 'models/crop_pred_final.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading the Crop Yield Recommendation model pickle file
pickled_model_file = open('models/yield_pred_final_1.pkl','rb')
classifier = pickle.load(pickled_model_file)


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None




app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Farmer Friend - Home'
    return render_template("index.html", title=title)

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Farmer Friend - Crop Recommendation'
    return render_template('crop.html', title=title)


@ app.route('/Yield-recommend')
def yield_recommend():
    title = 'Farmer Friend - Yield Prediction'

    return render_template('yield.html', title=title)

# render crop recommendation result page


@ app.route('/Yield-Predict', methods=['POST'])
def yield_prediction():
    title = 'Farmer Friend - Yield Recommendation'

    if request.method == 'POST':
        R = float(request.form['Rainfall'])
        A = float(request.form['Area'])
        T = float(request.form['Temp'])
        H = float(request.form['Humidity'])
        d9 = [A,H,R,T]
        h = [ "Area","Humidity","Rainfall","Temp", ]
        ra = pd.DataFrame(d9,index = h)



        pl = str(request.form['Crop'])
        crop = [ 'Banana', 'Blackgram', 'Coffee', 'Cotton', 'Grapes', 'Jute', 'Lentil', 'Maize', 'Mango', 'Orange', 'Papaya', 'PigeonPeas', 'Pomegranate', 'Rice']
        # x = input("Enter state")
        if  pl in crop:
            c1 = np.zeros(len(crop))
            c1[crop.index(pl)] = 1
        crp = pd.DataFrame(c1.tolist(),index =crop)




        S = str(request.form['Season'])
        season = ["Kharif", "Rabi", "Summer", "Whole Year", "Winter"]
         #x = input("Enter Season")
        if  S in season:
            s1 = np.zeros(len(season))
            s1[season.index(S)] = 1
        sea = pd.DataFrame(s1.tolist(),index =season)
          


        dt = str(request.form['District'])
        dist = ['24 PARAGANAS SOUTH', 'ADILABAD', 'AGAR MALWA', 'AGRA', 'AHMADABAD', 'AHMEDNAGAR', 'AIZAWL', 'AJMER', 'AKOLA', 'ALAPPUZHA', 'ALIGARH', 'ALIRAJPUR', 'ALLAHABAD', 'ALMORA', 'ALWAR', 'AMBALA', 'AMBEDKAR NAGAR', 'AMETHI', 'AMRAVATI', 'AMRELI', 'AMRITSAR', 'AMROHA', 'ANAND', 'ANANTAPUR', 'ANANTNAG', 'ANJAW', 'ANUGUL', 'ANUPPUR', 'ARARIA', 'ARIYALUR', 'ARWAL', 'ASHOKNAGAR', 'AURAIYA', 'AURANGABAD', 'AZAMGARH', 'BADGAM', 'BAGALKOT', 'BAGESHWAR', 'BAGHPAT', 'BAHRAICH', 'BAKSA', 'BALAGHAT', 'BALANGIR', 'BALESHWAR', 'BALLIA', 'BALOD', 'BALODA BAZAR', 'BALRAMPUR', 'BANAS KANTHA', 'BANDA', 'BANDIPORA', 'BANGALORE RURAL', 'BANKA', 'BANKURA', 'BANSWARA', 'BARABANKI', 'BARAMULLA', 'BARAN', 'BARDHAMAN', 'BAREILLY', 'BARGARH', 'BARMER', 'BARNALA', 'BARPETA', 'BARWANI', 'BASTAR', 'BASTI', 'BATHINDA', 'BEED', 'BEGUSARAI', 'BELGAUM', 'BELLARY', 'BEMETARA', 'BENGALURU URBAN', 'BETUL', 'BHADRAK', 'BHAGALPUR', 'BHANDARA', 'BHARATPUR', 'BHARUCH', 'BHAVNAGAR', 'BHILWARA', 'BHIND', 'BHIWANI', 'BHOJPUR', 'BHOPAL', 'BIDAR', 'BIJAPUR', 'BIJNOR', 'BIKANER', 'BILASPUR', 'BIRBHUM', 'BISHNUPUR', 'BOKARO', 'BONGAIGAON', 'BOUDH', 'BUDAUN', 'BULANDSHAHR', 'BULDHANA', 'BUNDI', 'BURHANPUR', 'BUXAR', 'CACHAR', 'CHAMARAJANAGAR', 'CHAMBA', 'CHAMOLI', 'CHAMPAWAT', 'CHAMPHAI', 'CHANDAULI', 'CHANDEL', 'CHANDIGARH', 'CHANDRAPUR', 'CHANGLANG', 'CHATRA', 'CHHATARPUR', 'CHHINDWARA', 'CHIKBALLAPUR', 'CHIKMAGALUR', 'CHIRANG', 'CHITRADURGA', 'CHITRAKOOT', 'CHITTOOR', 'CHITTORGARH', 'CHURACHANDPUR', 'CHURU', 'COIMBATORE', 'COOCHBEHAR', 'CUDDALORE', 'CUTTACK', 'DADRA AND NAGAR HAVELI', 'DAKSHIN KANNAD', 'DAMOH', 'DANG', 'DANTEWADA', 'DARBHANGA', 'DARJEELING', 'DARRANG', 'DATIA', 'DAUSA', 'DAVANGERE', 'DEHRADUN', 'DEOGARH', 'DEOGHAR', 'DEORIA', 'DEWAS', 'DHALAI', 'DHAMTARI', 'DHANBAD', 'DHAR', 'DHARMAPURI', 'DHARWAD', 'DHEMAJI', 'DHENKANAL', 'DHOLPUR', 'DHUBRI', 'DHULE', 'DIBANG VALLEY', 'DIBRUGARH', 'DIMA HASAO', 'DIMAPUR', 'DINAJPUR DAKSHIN', 'DINAJPUR UTTAR', 'DINDIGUL', 'DINDORI', 'DODA', 'DOHAD', 'DUMKA', 'DUNGARPUR', 'DURG', 'EAST DISTRICT', 'EAST GARO HILLS', 'EAST GODAVARI', 'EAST JAINTIA HILLS', 'EAST KAMENG', 'EAST KHASI HILLS', 'EAST SIANG', 'EAST SINGHBUM', 'ERNAKULAM', 'ERODE', 'ETAH', 'ETAWAH', 'FAIZABAD', 'FARIDABAD', 'FARIDKOT', 'FARRUKHABAD', 'FATEHABAD', 'FATEHGARH SAHIB', 'FATEHPUR', 'FAZILKA', 'FIROZABAD', 'FIROZEPUR', 'GADAG', 'GADCHIROLI', 'GAJAPATI', 'GANDERBAL', 'GANDHINAGAR', 'GANGANAGAR', 'GANJAM', 'GARHWA', 'GARIYABAND', 'GAUTAM BUDDHA NAGAR', 'GAYA', 'GHAZIABAD', 'GHAZIPUR', 'GIRIDIH', 'GOALPARA', 'GODDA', 'GOLAGHAT', 'GOMATI', 'GONDA', 'GONDIA', 'GOPALGANJ', 'GORAKHPUR', 'GULBARGA', 'GUMLA', 'GUNA', 'GUNTUR', 'GURDASPUR', 'GURGAON', 'GWALIOR', 'HAILAKANDI', 'HAMIRPUR', 'HANUMANGARH', 'HAPUR', 'HARDA', 'HARDOI', 'HARIDWAR', 'HASSAN', 'HATHRAS', 'HAVERI', 'HAZARIBAGH', 'HINGOLI', 'HISAR', 'HOOGHLY', 'HOSHANGABAD', 'HOSHIARPUR', 'HOWRAH', 'HYDERABAD', 'IDUKKI', 'IMPHAL EAST', 'IMPHAL WEST', 'INDORE', 'JABALPUR', 'JAGATSINGHAPUR', 'JAIPUR', 'JAISALMER', 'JAJAPUR', 'JALANDHAR', 'JALAUN', 'JALGAON', 'JALNA', 'JALORE', 'JALPAIGURI', 'JAMMU', 'JAMNAGAR', 'JAMTARA', 'JAMUI', 'JANJGIR-CHAMPA', 'JASHPUR', 'JAUNPUR', 'JEHANABAD', 'JHABUA', 'JHAJJAR', 'JHALAWAR', 'JHANSI', 'JHARSUGUDA', 'JHUNJHUNU', 'JIND', 'JODHPUR', 'JORHAT', 'JUNAGADH', 'KABIRDHAM', 'KACHCHH', 'KADAPA', 'KAIMUR (BHABUA)', 'KAITHAL', 'KALAHANDI', 'KAMRUP', 'KAMRUP METRO', 'KANCHIPURAM', 'KANDHAMAL', 'KANGRA', 'KANKER', 'KANNAUJ', 'KANNIYAKUMARI', 'KANNUR', 'KANPUR DEHAT', 'KANPUR NAGAR', 'KAPURTHALA', 'KARAIKAL', 'KARAULI', 'KARBI ANGLONG', 'KARGIL', 'KARIMGANJ', 'KARIMNAGAR', 'KARNAL', 'KARUR', 'KASARAGOD', 'KASGANJ', 'KATHUA', 'KATIHAR', 'KATNI', 'KAUSHAMBI', 'KENDRAPARA', 'KENDUJHAR', 'KHAGARIA', 'KHAMMAM', 'KHANDWA', 'KHARGONE', 'KHEDA', 'KHERI', 'KHORDHA', 'KHOWAI', 'KHUNTI', 'KINNAUR', 'KIPHIRE', 'KISHANGANJ', 'KISHTWAR', 'KODAGU', 'KODERMA', 'KOHIMA', 'KOKRAJHAR', 'KOLAR', 'KOLASIB', 'KOLHAPUR', 'KOLLAM', 'KONDAGAON', 'KOPPAL', 'KORAPUT', 'KORBA', 'KOREA', 'KOTA', 'KOTTAYAM', 'KOZHIKODE', 'KRISHNA', 'KRISHNAGIRI', 'KULGAM', 'KULLU', 'KUPWARA', 'KURNOOL', 'KURUKSHETRA', 'KURUNG KUMEY', 'KUSHI NAGAR', 'LAHUL AND SPITI', 'LAKHIMPUR', 'LAKHISARAI', 'LALITPUR', 'LATEHAR', 'LATUR', 'LAWNGTLAI', 'LOHARDAGA', 'LOHIT', 'LONGDING', 'LONGLENG', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI', 'LUCKNOW', 'LUDHIANA', 'LUNGLEI', 'MADHEPURA', 'MADHUBANI', 'MADURAI', 'MAHARAJGANJ', 'MAHASAMUND', 'MAHBUBNAGAR', 'MAHE', 'MAHENDRAGARH', 'MAHESANA', 'MAHOBA', 'MAINPURI', 'MALAPPURAM', 'MALDAH', 'MALKANGIRI', 'MAMIT', 'MANDI', 'MANDLA', 'MANDSAUR', 'MANDYA', 'MANSA', 'MARIGAON', 'MATHURA', 'MAU', 'MAYURBHANJ', 'MEDAK', 'MEDINIPUR EAST', 'MEDINIPUR WEST', 'MEERUT', 'MEWAT', 'MIRZAPUR', 'MOGA', 'MOKOKCHUNG', 'MON', 'MORADABAD', 'MORENA', 'MUKTSAR', 'MUMBAI', 'MUNGELI', 'MUNGER', 'MURSHIDABAD', 'MUZAFFARNAGAR', 'MUZAFFARPUR', 'MYSORE', 'NABARANGPUR', 'NADIA', 'NAGAON', 'NAGAPATTINAM', 'NAGAUR', 'NAGPUR', 'NAINITAL', 'NALANDA', 'NALBARI', 'NALGONDA', 'NAMAKKAL', 'NANDED', 'NANDURBAR', 'NARAYANPUR', 'NARMADA', 'NARSINGHPUR', 'NASHIK', 'NAVSARI', 'NAWADA', 'NAWANSHAHR', 'NAYAGARH', 'NEEMUCH', 'NICOBARS', 'NIZAMABAD', 'NORTH AND MIDDLE ANDAMAN', 'NORTH DISTRICT', 'NORTH GARO HILLS', 'NORTH GOA', 'NORTH TRIPURA', 'NUAPADA', 'OSMANABAD', 'PAKUR', 'PALAKKAD', 'PALAMU', 'PALGHAR', 'PALI', 'PALWAL', 'PANCH MAHALS', 'PANCHKULA', 'PANIPAT', 'PANNA', 'PAPUM PARE', 'PARBHANI', 'PASHCHIM CHAMPARAN', 'PATAN', 'PATHANAMTHITTA', 'PATHANKOT', 'PATIALA', 'PATNA', 'PAURI GARHWAL', 'PERAMBALUR', 'PEREN', 'PHEK', 'PILIBHIT', 'PITHORAGARH', 'PONDICHERRY', 'POONCH', 'PORBANDAR', 'PRAKASAM', 'PRATAPGARH', 'PUDUKKOTTAI', 'PULWAMA', 'PUNE', 'PURBI CHAMPARAN', 'PURI', 'PURNIA', 'PURULIA', 'RAE BARELI', 'RAICHUR', 'RAIGAD', 'RAIGARH', 'RAIPUR', 'RAISEN', 'RAJAURI', 'RAJGARH', 'RAJKOT', 'RAJNANDGAON', 'RAJSAMAND', 'RAMANAGARA', 'RAMANATHAPURAM', 'RAMBAN', 'RAMGARH', 'RAMPUR', 'RANCHI', 'RANGAREDDI', 'RATLAM', 'RATNAGIRI', 'RAYAGADA', 'REASI', 'REWA', 'REWARI', 'RI BHOI', 'ROHTAK', 'ROHTAS', 'RUDRA PRAYAG', 'RUPNAGAR', 'S.A.S NAGAR', 'SABAR KANTHA', 'SAGAR', 'SAHARANPUR', 'SAHARSA', 'SAHEBGANJ', 'SAIHA', 'SALEM', 'SAMASTIPUR', 'SAMBA', 'SAMBALPUR', 'SAMBHAL', 'SANGLI', 'SANGRUR', 'SANT KABEER NAGAR', 'SANT RAVIDAS NAGAR', 'SARAIKELA KHARSAWAN', 'SARAN', 'SATARA', 'SATNA', 'SAWAI MADHOPUR', 'SEHORE', 'SENAPATI', 'SEONI', 'SEPAHIJALA', 'SERCHHIP', 'SHAHDOL', 'SHAHJAHANPUR', 'SHAJAPUR', 'SHAMLI', 'SHEIKHPURA', 'SHEOHAR', 'SHEOPUR', 'SHIMLA', 'SHIMOGA', 'SHIVPURI', 'SHOPIAN', 'SHRAVASTI', 'SIDDHARTH NAGAR', 'SIDHI', 'SIKAR', 'SIMDEGA', 'SINDHUDURG', 'SINGRAULI', 'SIRMAUR', 'SIROHI', 'SIRSA', 'SITAMARHI', 'SITAPUR', 'SIVAGANGA', 'SIVASAGAR', 'SIWAN', 'SOLAN', 'SOLAPUR', 'SONBHADRA', 'SONEPUR', 'SONIPAT', 'SONITPUR', 'SOUTH ANDAMANS', 'SOUTH DISTRICT', 'SOUTH GARO HILLS', 'SOUTH GOA', 'SOUTH TRIPURA', 'SOUTH WEST GARO HILLS', 'SOUTH WEST KHASI HILLS', 'SPSR NELLORE', 'SRIKAKULAM', 'SRINAGAR', 'SUKMA', 'SULTANPUR', 'SUNDARGARH', 'SUPAUL', 'SURAJPUR', 'SURAT', 'SURENDRANAGAR', 'SURGUJA', 'TAMENGLONG', 'TAPI', 'TARN TARAN', 'TAWANG', 'TEHRI GARHWAL', 'THANE', 'THANJAVUR', 'THE NILGIRIS', 'THENI', 'THIRUVALLUR', 'THIRUVANANTHAPURAM', 'THIRUVARUR', 'THOUBAL', 'THRISSUR', 'TIKAMGARH', 'TINSUKIA', 'TIRAP', 'TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPPUR', 'TIRUVANNAMALAI', 'TONK', 'TUENSANG', 'TUMKUR', 'TUTICORIN', 'UDAIPUR', 'UDALGURI', 'UDAM SINGH NAGAR', 'UDHAMPUR', 'UDUPI', 'UJJAIN', 'UKHRUL', 'UMARIA', 'UNA', 'UNAKOTI', 'UNNAO', 'UPPER SIANG', 'UPPER SUBANSIRI', 'UTTAR KANNAD', 'UTTAR KASHI', 'VADODARA', 'VAISHALI', 'VALSAD', 'VARANASI', 'VELLORE', 'VIDISHA', 'VILLUPURAM', 'VIRUDHUNAGAR', 'VISAKHAPATANAM', 'VIZIANAGARAM', 'WARANGAL', 'WARDHA', 'WASHIM', 'WAYANAD', 'WEST DISTRICT', 'WEST GARO HILLS', 'WEST GODAVARI', 'WEST JAINTIA HILLS', 'WEST KAMENG', 'WEST KHASI HILLS', 'WEST SIANG', 'WEST SINGHBHUM', 'WEST TRIPURA', 'WOKHA', 'YADGIR', 'YAMUNANAGAR', 'YANAM', 'YAVATMAL', 'ZUNHEBOTO']
         # x = input("Enter state")
        if  dt in dist:
            d1 = np.zeros(len(dist))
            d1[dist.index(dt)] = 1
        dst = pd.DataFrame(d1.tolist(),index =dist)       
 



        sta = str(request.form['State'])
        states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
        # x = input("Enter state")
        if  sta in states:
            s2 = np.zeros(len(states))
            s2[states.index(sta)] = 1
        st = pd.DataFrame(s2.tolist(),index =states)


        dummy = pd.concat([ra,crp,sea,dst,st])
            #data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = classifier.predict((dummy).T)
        final_prediction = "{:.3f}".format(my_prediction[0])

        return render_template('yield-result.html', prediction=final_prediction, title=title)
    else:
        return render_template('try_again.html', title=title)

# Yield Prediction Result page

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Farmer Friend - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['ph'])
        humidity = float(request.form['rainfall'])

        # state = request.form.get("stt")
     #   city = request.form.get("city")

     #   if weather_fetch(city) != None:
      #      temperature, humidity = weather_fetch(city)
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, title=title)

    else:

        return render_template('try_again.html', title=title)








if __name__ == "__main__":
    app.run(debug = True)