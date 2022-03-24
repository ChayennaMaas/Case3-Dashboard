#!/usr/bin/env python
# coding: utf-8

# # Case 3 - Laadpalen en elektrische auto data!

# Groep 2: Anna de Geeter, Daan Handgraaf, Chayenna Maas, Iris van de Velde

# # Inspecteer datasets

# In[1]:


import json
import requests
import pandas as pd
import folium
import plotly.express as px
import datetime
import plotly.graph_objects as go
import statsmodels.api as sm


# In[2]:


# pip install streamlit-folium


# In[3]:


#Inladen API - kijk naar country code en maxresults
response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=100&compact=true&verbose=false&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")


# In[4]:


#Omzetten naar dictionary
repsonsejson  = response.json()


# In[5]:


#Dataframe bevat kolom die een list zijn. 
#Met json_normalize zet je de eerste kolom om naar losse kolommen
Laadpalen = pd.json_normalize(repsonsejson)
#Daarna nog handmatig kijken welke kolommen over zijn in dit geval Connections
#Kijken naar eerst laadpaal op de locatie
#Kan je uitpakken middels:
df4 = pd.json_normalize(Laadpalen.Connections)
df5 = pd.json_normalize(df4[0])
df5.head()
###Bestanden samenvoegen
OCM = pd.concat([Laadpalen, df5], axis=1)


# In[6]:


# #Importeer OpenChargeMap
# url = "https://api.openchargemap.io/v3/poi"

# querystring = {"output":"json","countrycode":"NL","distanceunit":"km","verbose":"false","compact":"true","camelcase":"true","key":"5a180888-22fc-4c88-8e49-2390ae0422d3"}
# headers = {"Content-Type": "application/json"}

# response = requests.request("GET", url, headers=headers, params=querystring)

# #print en inspecteer de response 
# print(response.json())


# In[7]:


# #Prepareren tot een dataframe
# OCM_norm=pd.json_normalize(response.json())

# #Kolom 'connections' verder uitpakken
# #drop de kolom 'connections' van OCM_norm dataframe connections komt hiervoor in de plaats
# connections=pd.json_normalize(response.json(), record_path='connections')
# OCM_norm=OCM_norm.drop(labels='connections', axis=1)

# #Bestanden samenvoegen en print dataframe
# OCM=pd.concat([OCM_norm, connections], axis=1)
# # OCM
# OCM


# In[8]:


#Importeer: Laappaal data
laadpalen_gebruik=pd.read_csv('laadpalen_gebruik_v3.csv')


# In[9]:


#Importeer: Gekentekende_voertuigen
inputs=requests.get('https://opendata.rdw.nl/resource/m9d7-ebf2.json?$limit=200000')
response=inputs.json()

#Omzetten naar dataframe
gekentekende_voer=pd.DataFrame.from_dict(response)


# In[10]:


#Importeer: Verschillende brandstof auto's
inputs2=requests.get('https://opendata.rdw.nl/resource/8ys7-d773.json?$limit=1000000')
response2=inputs2.json()

#Omzetten naar dataframe
gekentekende_voer_brandstof=pd.DataFrame.from_dict(response2)


# ## Data Cleaning

# ### OCM

# In[11]:


OCM.describe()


# In[12]:


OCM.isna().sum().sort_values(ascending=False)


# In[13]:


# OCM=OCM.drop(labels=['AddressInfo.ContactEmail',' Comments', 'AddressInfo.RelatedURL', 
#                      'AddressInfo.AddressLine2', 'AddressInfo.ContactTelephone1', 
#                      'AddressInfo.AccessComments', 'Reference, OperatorsReference', 
#                      'GeneralComments', 'DateLastVerified', 'UsageCost'], axis=1)


# In[14]:


# OCM.isna().sum().sort_values(ascending=False)


# In[15]:


OCM['AddressInfo.StateOrProvince'].unique()


# In[16]:


OCM.loc[OCM.index ==99, ['AddressInfo.StateOrProvince']] = 'Gelderland'
OCM.loc[OCM.index ==97, ['AddressInfo.StateOrProvince']] = 'Noord-Holland'
OCM.loc[OCM.index ==95, ['AddressInfo.StateOrProvince']] = 'Zeeland'
OCM.loc[OCM.index ==92, ['AddressInfo.StateOrProvince']] = 'Zuid-Holland'
OCM.loc[OCM.index ==87, ['AddressInfo.StateOrProvince']] = 'Noord-Brabant'
OCM.loc[OCM.index ==83, ['AddressInfo.StateOrProvince']] = 'Groningen'
OCM.loc[OCM.index ==78, ['AddressInfo.StateOrProvince']] = 'Overijssel'
OCM.loc[OCM.index ==75, ['AddressInfo.StateOrProvince']] = 'Utrecht'
OCM.loc[OCM.index ==74, ['AddressInfo.StateOrProvince']] = 'Utrecht'
OCM.loc[OCM.index ==67, ['AddressInfo.StateOrProvince']] = 'Drenthe'
OCM.loc[OCM.index ==64, ['AddressInfo.StateOrProvince']] = 'Gelderland'
OCM.loc[OCM.index ==63, ['AddressInfo.StateOrProvince']] = 'Utrecht'
OCM.loc[OCM.index ==62, ['AddressInfo.StateOrProvince']] = 'Zuid-Holland'
OCM.loc[OCM.index ==61, ['AddressInfo.StateOrProvince']] = 'Noord-Brabant'
OCM.loc[OCM.index ==60, ['AddressInfo.StateOrProvince']] = 'Zuid-Holland'
OCM.loc[OCM.index ==58, ['AddressInfo.StateOrProvince']] = 'Overijssel'
OCM.loc[OCM.index ==48, ['AddressInfo.StateOrProvince']] = 'Noord-Holland'
OCM.loc[OCM.index ==65, ['AddressInfo.StateOrProvince']] = 'Noord-Brabant'


# In[17]:


OCM['AddressInfo.StateOrProvince'].value_counts().sort_values(ascending = False)


# In[18]:


OCM= OCM.replace(to_replace=['North Holland','NH', 'North-Holland'], value='Noord-Holland')
OCM= OCM.replace(to_replace='North Brabant', value='Noord-Brabant')
OCM= OCM.replace(to_replace='South Holland', value='Zuid-Holland')
OCM= OCM.replace(to_replace=['UT','UTRECHT'], value='Utrecht')


# In[19]:


OCM['AddressInfo.StateOrProvince'].value_counts().sort_values(ascending = False)


# ## Laadpalen locatie: Kaart

# In[20]:


def color_producer(type):
   if type =='Utrecht':
        return 'gold'
   elif type == 'Noord-Holland':
        return 'pink'
   elif type == 'Zuid-Holland':
        return 'blue'
   elif type == 'Noord-Brabant':
        return 'orange'
   elif type == 'Gelderland':
        return 'darkviolet'
   elif type == 'Overijssel':
        return 'salmon'
   elif type == 'Groningen':
        return 'lawngreen'
   elif type == 'Zeeland':
        return 'fuchsia'
   elif type == 'Drenthe':
        return 'yellow'
   elif type == 'Limburg':
        return 'aqua'
   else: 
        return 'red'


# In[21]:


def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


# In[66]:


#Laad map in
from streamlit_folium import folium_static

Laadpalen_loc = folium.Map(location=[52.3702157,4.8951679],zoom_start=7,tiles="cartodbpositron")

locations = OCM[['AddressInfo.Latitude','AddressInfo.Longitude']]
locationlist = locations.values.tolist()
len(locationlist)

#Voeg locatie punten aan map toe
lat=OCM['AddressInfo.Latitude']
lon=OCM['AddressInfo.Longitude']
types=OCM['AddressInfo.StateOrProvince']
loc=OCM['AddressInfo.StateOrProvince']
for lt, ln, el, x in zip(lat, lon, types, loc):
    cm = folium.CircleMarker(location=[lt, ln],
                            radius = 4,
                            fill=True,
                            fill_color=color_producer(el),
                            color = False,
                            fill_opacity=1).add_to(Laadpalen_loc)

#Pop ups aanmaken
for x in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[x], popup=OCM['AddressInfo.AddressLine1'][x], 
                        radius = 4,fill=True, fill_color=None, color= False, fill_opacity=0.0).add_to(Laadpalen_loc)

#Voeg legenda aan kaart toe
Laadpalen_loc=add_categorical_legend(Laadpalen_loc, 'Provincie(s)',
                             colors = ['gold', 'pink', 'blue', 'orange', 'darkviolet', 'salmon', 'lawngreen', 'fuchsia',
                                      'yellow', 'aqua', 'red'],
                           labels = ['Utrecht', 'Noord-Holland','Zuid-Holland','Noord-Brabant','Gelderland','Overijssel',
                                     'Groningen', 'Zeeland', 'Drenthe','Limburg ', 'Friesland '])

# Laadpalen_loc


# ## Laadpalen

# In[23]:


laadpalen_gebruik.isna().sum().sort_values(ascending=False)


# In[24]:


laadpalen_gebruik.describe()


# ## Kenteken

# In[25]:


gekentekende_voer_combined=gekentekende_voer.merge(gekentekende_voer_brandstof, on='kenteken', how='inner')


# In[26]:


gekentekende_voer_combined.isna().sum().sort_values(ascending=False)


# In[27]:


gekentekende_voer_combined=gekentekende_voer_combined.drop(labels=['maximum_last_onder_de_vooras_sen_tezamen_koppeling','afwijkende_maximum_snelheid','aantal_staanplaatsen',
                                                                  'wielbasis_voertuig_minimum','uitstoot_deeltjes_zwaar','wielbasis_voertuig_maximum','max_vermogen_15_minuten',
                                                                  'breedte_voertuig_maximum','type_gasinstallatie','lengte_voertuig_minimum','massa_bedrijfsklaar_maximaal',
                                                                  'massa_bedrijfsklaar_minimaal','breedte_voertuig_minimum','hoogte_voertuig_minimum','hoogte_voertuig_maximum',
                                                                  'lengte_voertuig_maximum','oplegger_geremd','vervaldatum_tachograaf','vervaldatum_tachograaf_dt', 'milieuklasse_eg_goedkeuring_zwaar',
                                                                  'actie_radius_extern_opladen_stad_wltp','aanhangwagen_autonoom_geremd','actie_radius_extern_opladen_wltp','elektrisch_verbruik_extern_opladen_wltp',
                                                                  'brandstof_verbruik_gewogen_gecombineerd_wltp', 'emis_co2_gewogen_gecombineerd_wltp', 'co2_uitstoot_gewogen', 'maximale_constructiesnelheid_brom_snorfiets',          
                                                                   'europese_uitvoeringcategorie_toevoeging','europese_voertuigcategorie_toevoeging','max_vermogen_60_minuten', 'aanhangwagen_middenas_geremd', 'plaats_chassisnummer', 
                                                                   'uitstoot_deeltjes_licht','laadvermogen','roetuitstoot','actie_radius_enkel_elektrisch_stad_wltp','elektrisch_verbruik_enkel_elektrisch_wltp','actie_radius_enkel_elektrisch_wltp',
                                                                  'emis_deeltjes_type1_wltp','subcategorie_nederland','nominaal_continu_maximumvermogen','brandstof_verbruik_gecombineerd_wltp','klasse_hybride_elektrisch_voertuig',
                                                                  'netto_max_vermogen_elektrisch','emissie_co2_gecombineerd_wltp','brandstofverbruik_stad','brandstofverbruik_buiten','zuinigheidsclassificatie','verticale_belasting_koppelpunt_getrokken_voertuig',
                                                                   'brandstofverbruik_gecombineerd','co2_uitstoot_gecombineerd','maximum_trekken_massa_geremd','opgegeven_maximum_snelheid', 'toerental_geluidsniveau','geluidsniveau_stationair',
                                                                   'nettomaximumvermogen','maximum_massa_trekken_ongeremd','hoogte_voertuig', 'bruto_bpm','uitlaatemissieniveau','geluidsniveau_rijdend',
                                                                   'milieuklasse_eg_goedkeuring_licht','maximum_ondersteunende_snelheid','aantal_wielen','aantal_zitplaatsen','wielbasis'], axis=1)


# In[28]:


gekentekende_voer_combined.isna().sum().sort_values(ascending=False)


# ### Line plot

# In[29]:


gekentekende_voer_combined['datum_tenaamstelling']=pd.to_datetime(gekentekende_voer_combined['datum_tenaamstelling'])


# In[30]:


voer_per_brandstof=gekentekende_voer_combined[['kenteken','brandstof_omschrijving', 'datum_tenaamstelling']]


# In[31]:


gekentekende_voer_combined['brandstof_omschrijving'].unique()


# In[32]:


#split dataframe voor benzine
benzine=voer_per_brandstof[voer_per_brandstof['brandstof_omschrijving']=='Benzine']


# In[33]:


#Sorteer op datum 
benzine=benzine.sort_values(by='datum_tenaamstelling')
benzine['cumhelp']=1
benzine['jaar']=benzine['datum_tenaamstelling'].dt.year
benzine['cumulative']=benzine['cumhelp'].cumsum()


# In[34]:


#split dataframe voor elektriciteit
elektriciteit=voer_per_brandstof[voer_per_brandstof['brandstof_omschrijving']=='Elektriciteit']


# In[35]:


#Sorteer op datum 
elektriciteit=elektriciteit.sort_values(by='datum_tenaamstelling')
elektriciteit['cumhelp']=1
elektriciteit['jaar']=elektriciteit['datum_tenaamstelling'].dt.year
elektriciteit['cumulative']=elektriciteit['cumhelp'].cumsum()


# In[36]:


#split dataframe voor diesel
diesel=voer_per_brandstof[voer_per_brandstof['brandstof_omschrijving']=='Diesel']


# In[37]:


#Sorteer op datum 
diesel=diesel.sort_values(by='datum_tenaamstelling')
diesel['cumhelp']=1
diesel['jaar']=diesel['datum_tenaamstelling'].dt.year
diesel['cumulative']=diesel['cumhelp'].cumsum()


# In[38]:


#split dataframe voor lpg
lpg=voer_per_brandstof[voer_per_brandstof['brandstof_omschrijving']=='LPG']


# In[39]:


#Sorteer op datum
lpg=lpg.sort_values(by='datum_tenaamstelling')
lpg['cumhelp']=1
lpg['jaar']=lpg['datum_tenaamstelling'].dt.year
lpg['cumulative']=lpg['cumhelp'].cumsum()


# In[40]:


#split dataframe voor cng
waterstof=voer_per_brandstof[voer_per_brandstof['brandstof_omschrijving']=='waterstof']


# In[41]:


#Sorteer op datum
waterstof=waterstof.sort_values(by='datum_tenaamstelling')
waterstof['cumhelp']=1
waterstof['jaar']=waterstof['datum_tenaamstelling'].dt.year
waterstof['cumulative']=waterstof['cumhelp'].cumsum()


# In[42]:


#combineer dataframes tot 1
brandstof_ver=pd.concat([waterstof, lpg, diesel, benzine, elektriciteit])
brandstof_ver_2021=brandstof_ver[brandstof_ver['jaar']==2021]


# In[65]:


#specifeer de kleuren
ind_color_map={'LPG':'rgb(75,196,213)', 'Diesel': 'rgb(234,62,112)', 'Benzine': 'rgb(255,211,92)', 
               'Elektriciteit':'rgb(128,0,128)'}

#plot de line plot per categorie brandstof
cum_brand=px.line(brandstof_ver, x='datum_tenaamstelling',y='cumulative', color_discrete_map=ind_color_map, color='brandstof_omschrijving',
            labels={'brandstof_omschrijving':'Brandstof categorie', 'cumulative': '''Aantal auto's''', 'datum_tenaamstelling':'Tijd'},
           title='''<b>Cumulatief aantal auto's per brandstofcategorie<b>''', height=800)

cum_brand.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="year",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        )
    ))

cum_brand.update_xaxes(nticks=12)

# cum_brand.show()


# ## Laadpalen hist

# In[44]:


#Verwijder de outliers
laadpalen_gebruik = laadpalen_gebruik[laadpalen_gebruik['ChargeTime'] >= 0]
laadpalen_gebruik.describe()


# In[64]:


#Maak histogram aan
fig3= px.histogram(laadpalen_gebruik, x=['ChargeTime','ConnectedTime'],
                   barmode='overlay', nbins=200,color_discrete_sequence=['orange', 'lightblue'], 
                   opacity=0.6, range_x=[0,20],marginal='box', labels={'variable':'Variabel'})

#Update de de layout
fig3.update_layout(title='<b>Laadpalengebruik: Laad tijd & Aansluit tijd<b>',xaxis_title='Aantal uren', 
                   yaxis_title='Aantal klanten', height=800)
fig3.update_xaxes(nticks=15)

#Voeg notaties toe
fig3.add_annotation(x=16.1, y=2900,
            text="<b>Gemiddelde laadtijd: 2.49 uur<b>",
            showarrow=False)
fig3.add_annotation(x=15.8, y=2700,
            text="<b>Mediaan laadtijd: 2.23 uur<b>",
            showarrow=False)
fig3.add_annotation(x=16.5, y=2500,
            text="<b>Gemiddelde ConnectedTime: 6.35<b>",
            showarrow=False)
fig3.add_annotation(x=16.2, y=2300,
            text="<b>Mediaan ConnectedTime: 3.80<b>",
            showarrow=False)

fig3.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="hour",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        )
    ))


# fig3.show()


# ## Welke merken zijn hot?

# In[46]:


#Hernoem de data
gekentekende_voer_combined= gekentekende_voer_combined.replace(to_replace=['VW'], value='VOLKSWAGEN')
gekentekende_voer_combined= gekentekende_voer_combined.replace(to_replace=['BMW I'], value='BMW')
gekentekende_voer_combined= gekentekende_voer_combined.replace(to_replace=['MERCEDES-AMG'], value='MERCEDES-BENZ')
gekentekende_voer_combined= gekentekende_voer_combined.replace(to_replace=['TESLA MOTORS'], value='TESLA')


# In[47]:


#Filter de dataset op alleen elektrische personenauto's
personenauto=gekentekende_voer_combined[gekentekende_voer_combined['voertuigsoort']=='Personenauto']
elektrisch = personenauto[personenauto['brandstof_omschrijving']=='Elektriciteit']


# In[63]:


#Maak barplot van de data
merk_hot=px.bar(elektrisch,x='merk')
merk_hot.update_layout(title_text='<b>Automerken: elektrisch<b>')
merk_hot.update_xaxes(tickangle=45)
#Laat plot zien
# merk_hot.show()


# In[49]:


# pie_chart = px.pie(elektrisch, names='merk', title='<b>Verdeling merken<b>')
# pie_chart.show()


# ## Verhouding elektrische auto's vs totaal over tijd

# In[50]:


#Maak nieuwe kolom aan op m-j aan ter voorbereiding
brandstof_ver['M-Y']=brandstof_ver['datum_tenaamstelling'].dt.to_period('M')

# Sorteer dataframe op maand
data_sort=brandstof_ver.sort_values(by='M-Y')

# Zorg dat de rijeen index vanaf 0/1 en optellend hebben
data_sort_reset=data_sort.reset_index(drop=True)
data_sort_reset.index=data_sort_reset.index+1

# Filter dataframe met enkel verhouding == elektrisch
data_sort_elek=data_sort_reset[data_sort_reset['brandstof_omschrijving'] == 'Elektriciteit']

# Genereer nieuw colom die cumilative / (rij_index + 1) is
data_sort_elek['verhouding']=data_sort_elek['cumulative']/data_sort_elek.index


# In[51]:


#Zet data_sort_elek['M-Y'] om naar numerieke waardes
df = pd.get_dummies(data_sort_elek, columns=['M-Y'], drop_first=True)

#setup voor linear regression d.m.v. sm.OLS
Y=df['verhouding']
datum=df.drop(labels=['kenteken','brandstof_omschrijving','datum_tenaamstelling','cumhelp','jaar',
                           'cumulative','verhouding'], axis=1)
X=datum
X=sm.add_constant(X)

trend = sm.OLS(Y,X).fit().fittedvalues


# In[62]:


# De berekende waarden zijn nu klaar om te plotten.
#Maak basis plot
elek_ver = px.scatter(x=df['datum_tenaamstelling'], y=df['verhouding'],labels={'x': 'Tijd', 'y':'Percentage'}
                 , title='''<b>Verhouding elektrische auto's vs totaal<b>''', height=800)

#Voeg de berekende waarde toe aan plot
elek_ver.add_traces(go.Scatter(x=df['datum_tenaamstelling'], y=trend,mode = 'lines', name='trendline'))


#Voeg een range slider toe voor de tijd
elek_ver.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="month",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        )
    ))

#show plot
# elek_ver.show()


# In[53]:


#defineer response variabel
y = df['verhouding']

#defineer predictor variabelen
datum=df.drop(labels=['kenteken','brandstof_omschrijving','datum_tenaamstelling','cumhelp','jaar',
                           'cumulative','verhouding'], axis=1)
x = datum

#voeg constante to aan predictor variables
x = sm.add_constant(x)

#fit regression model
model = sm.OLS(y, x).fit()

#Laat samenvatting van de model fit zien
print(model.summary())


# # Streamlit

# In[54]:


import streamlit as st


# In[55]:


st.title('''Dashboard: Elektrische auto's''')


# In[75]:


st.write('Auteurs: Anna de Geeter, Daan Handgraaf, Chayenna Maas, Iris van de Velde')


# In[57]:


st.subheader("Locatie laadpalen")


# In[67]:


col1, col2 = st.columns(2)

with col1:
    st.write('In de grafiek zijn 100 locaties van laadpalen te zien. '
            'Wat opvalt is dat de meeste laadpalen zich bevinden rondom '
           'de randstad. Wel valt te zeggen dat dit er maar 100 zijn '
           'en dat het moeilijk te zeggen is hoe de verdeling er uit zal zien, '
           'wanneer er meer locaties worden meegenomen')
    st.image('Schermafbeelding 2022-03-24 111829.png')

with col2:
    folium_static(Laadpalen_loc)


# In[59]:


st.subheader('Laadpalen gebruik')


# In[68]:


col1, col2 = st.columns(2)

with col1:
    st.write('In deze histogram zijn de volgende twee variabelen geplot: '
            'de laadtijd (ChargeTime) en de aansluitingstijd (ConnectedTime). '
            'Met de laadtijd wordt aangegeven hoe lang de auto bezig is met '
            'laden en met de aansluitingstijd hoe lang de auto aangesloten is '
            'op de laadpaal. Uit het histogram blijkt dat de meeste auto’s een '
            'chargetime hebben tussen de 1.5 uur en 2.5 uur. De gemiddelde laadtijd '
            'bedraagt 2.49 uur. De gemiddelde aansluitingstijd daarin tegen bedraagt '
            '6.35 uur. Dit betekend dat mensen hun auto veel langer aan de laadpaal '
            'aansluiten dan nodig.')
    st.write('In de histogram is dit ook te zien. De bin met de langste charge time '
            'tussen de 6.5 uur en 7.5 uur is. Er zijn 136 auto’s die een laadtijd hebben '
            'met deze tijd, maar er zijn in totaal 311 auto’s die zo lang aan de laadpaal '
            'waren aangesloten. Ook loopt de variabele van de aansluitingstijd verder in de '
            'histogram dan de laadtijd. Er is geen enkele auto met een laadtijd van boven de '
            '7.5 uur maar toch worden sommige auto’s tot wel 18.5 uur aan de laadpaal gelaten)')

with col2:
    st.write(fig3)


# In[72]:


st.subheader("Welke merken zijn hot?")


# In[69]:


col1, col2 = st.columns(2)

with col1:
    st.write('In de grafiek is te zien dat Kia er, in verhouding '
            'met de andere merken, bovenuit springt met het aantal '
            '''elektrische auto's. Kia komt meer dan 200 keer in de data''' 
            '''voor. Op de tweede plaats komt Audi, met ongeveer 170 auto's.''' 
            'Audi wordt gevolgd door twee merken die vrij gelijk op gaan: '
            'Hyundai en BMW. Beide komen deze automerken ongeveer 140 keer voor in de data.')

with col2:
    st.write(merk_hot)


# In[71]:


st.subheader('''Aantal auto's per brandstofcategorie''')


# In[70]:


col1, col2 = st.columns(2)

with col1:
    st.write('''In dese plot is het cumulatief van de aantal auto's ''' 
            'per brandstof categorie te zien. De vier verschillende '
            'brandstof categorieën zijn: LPG, Diesel, Benzine en '
            'Elektriciteit.')
    st.write('De verandering over tijd is goed te zien in dit plot. '
            'Zo kan je zien dat vanaf midden 2018 de auto’s met de '
            'brandstofcategorie ‘Elektriciteit’ toe neemt. '
            'Deze stijging word vanaf juli 2019 steeds steiler en '
            'het aantal auto’s met de brandstofcategorie ‘Elektriciteit’ '
            'neemt vanaf dat punt flink toe. Ook is er te zien dat het '
            'aantal auto’s met de brandstofcategorie ‘Benzine’ elk jaar '
            'het meest is. Het verschil tussen de aantal auto’s met de '
            'brandstofcategorie ‘Benzine’ en de brandstofcategorie '
            '‘Elektriciteit’ word door de loop van de jaren echter wel '
            'steeds kleiner. Het aantal auto’s met de brandstofcategorie '
            '‘Diesel’ word in de loop van de jaren ook ingehaald door het '
            'aantal elektrische auto’s.')

with col2:
    st.write(cum_brand)


# In[73]:


st.subheader('''Aantal auto's per brandstofcategorie''')


# In[74]:


col1, col2 = st.columns(2)

with col1:
    st.write('''In deze plot is het percentage van de elektrische auto's '''
            'over de jaren heen geplot. Om een voorspelling te maken over '
           'het gedrag is er met behulp van OLS regression een trendline aangebracht. '
           'Vervolgens is bij de samenvatting van de model fit te zien '
            'wat de R-squared is en wat de '
           'std err is per data punt en de coeff. Door deze data kan er een schatting '
           '''worden gemaakt hoe de verhouding van de elektische auto's t.o.v. de rest '''
           'zich zal gedragen in de toekomst')
    st.write('In de samenvatting is te zien dat de verhouding vanaf 2018 een '
           'vrij steil stijgt (zo rond de 10%), maar dat het vanaf okt 2021 '
            'een klein beetje afneemt. Het percentage stijgt nog steeds, maar minder '
           '''nu zo'n 3 tot 5 %. De verwachting is dat ook dat de elektrische auto's'''
            'nog steeds in toekomst zullen toenemen, maar dat dit per jaar zal afnemen')
    st.image('OLS2.png',use_column_width=True)

with col2:
    st.write(elek_ver)

