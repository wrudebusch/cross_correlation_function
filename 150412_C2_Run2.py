import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

## notes ## drx.index = 2015-04-12 12:40:33, ..., 2015-04-12 14:15:46
# no hnu raw data

## interpolation function
def fix(df):
    df.drop_duplicates(cols='datetime', take_last=True, inplace=True)
    #df = df.reset_index()
    start = df.datetime.iloc[0]
    end = df.datetime.iloc[-1]
    rng = pd.date_range(start, end, freq='S')
    df1 = df.set_index('datetime')
    df2 = df1.reindex(rng)
    #print df2.head()
    df3 =df2.convert_objects(convert_numeric=True)
    df4 = df3.apply(pd.Series.interpolate)
    #print df3.head()
    return(df4)

def normalize(A):
    A -= A.mean(); A /= A.std()
    return A

def maxccf(A,B):
    nsamples=len(A)
    xcorr = correlate(A, B)
    et = np.arange(1-nsamples, nsamples)
    recovered_time_shift = et[xcorr.argmax()]
    #print recovered_time_shift
    return  recovered_time_shift

def lag(df,lag=0):
    #df = df.set_index('datetime')
    df.index = df.index+pd.offsets.Second(lag)
    return df

## small functions to fix timestapms
dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S')
dateparse2 = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
dateparse3 = lambda x: pd.datetime.strptime(x, '%m/%d/%y %I:%M:%S %p') 
dateparse4 = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M %S')
dateparse5 = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S,')
dateparse6 = lambda x: pd.datetime.strptime(x, '%d-%b-%y %H:%M:%S')
dateparse7 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
dateparse8 = lambda x: pd.datetime.strptime(x, '%d-%b-%y %I:%M:%S %p')
dateparse9 = lambda x: pd.datetime.strptime(x, '%I:%M:%S %p') 

vim_eng_raw = pd.read_table('Engine_C2_vims_Run2.csv', sep=',', skiprows=5, header=None, parse_dates={'datetime': [0, 1]}, date_parser=dateparse6)
#print vim_eng_raw.head()

et_eng_raw = pd.read_table('Engine_C2_April12th2015_ET.csv', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse9)
et_eng_raw.datetime = et_eng_raw.datetime.map(lambda t: t.replace(year=2015, month=4, day=12))

lan_raw = pd.read_table('150412_C2_Langan_T15x_CO_Measurer_141268.csv', sep=',', skiprows=3, header=None, parse_dates={'datetime': [1]}, date_parser=dateparse3)
## check for blank lines at the end. 
ae52_raw = pd.read_table('AE52_C2_Run2.csv', sep=',', error_bad_lines=False, header=None, skiprows=2, parse_dates={'datetime': [0, 1]}, date_parser=dateparse6)

## part 1
drx_raw = pd.read_table('20150412143317_DRX.txt', skiprows=2, header=None, sep='\t', error_bad_lines=False, parse_dates={'datetime': [0]}, date_parser=dateparse2)
co2bg_raw = pd.read_table('20150412143317_CO2_BG.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)
co2dil_raw = pd.read_table('20150412143317_CO2_Dil.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)
co2tpi_raw = pd.read_table('20150412143317_CO2_TPi.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)
fl_raw = pd.read_table('20150412143317_Flowrate.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)
gps_raw = pd.read_table('20150412143317_GPS.txt', skiprows=3, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse5)
gps2_raw = pd.read_table('20150412143317_GPS2.txt', skiprows=3, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse5)
hnu_raw = pd.read_table('20150412143317_HNU.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)
testo_raw = pd.read_table('20150412143317_Testo350.txt', sep=',', skiprows=2, header=None, parse_dates={'datetime': [0]}, date_parser=dateparse2)


## fixing columns and names

vim_eng_raw = pd.DataFrame({'datetime' : vim_eng_raw['datetime'],
                    'vim_eng_fuel_rate' : vim_eng_raw[16],
                    'vim_eng_rpm' : vim_eng_raw[2],
                    'vim_eng_load' : vim_eng_raw[12],
                    'vim_eng_ground_spd' : vim_eng_raw[37],
                    'vim_eng_payload' : vim_eng_raw[34]})

#print vim_eng_raw.head()

et_eng_raw = pd.DataFrame({'datetime' : et_eng_raw['datetime'],
                    'et_eng_fuel_rate' : et_eng_raw[16],
                    'et_eng_rpm' : et_eng_raw[3],
                    'et_eng_load' : et_eng_raw[2],
                    'et_eng_ground_spd' : et_eng_raw[7]})

#print et_eng_raw.head()

gps_raw = pd.DataFrame({'datetime' : gps_raw['datetime'],
                    'gps_latitude' : gps_raw[3],
                    'gps_longitude' : gps_raw[5],
                    'gps_velocity' : gps_raw[7],
                    'gps_elevation' : gps_raw[8]})


gps2_raw = pd.DataFrame({'datetime' : gps2_raw['datetime'],
                    'gps2_latitude' : gps2_raw[2],
                    'gps2_longitude' : gps2_raw[4],
                    'gps2_velocity' : gps2_raw[8],
                    'gps2_elevation' : gps2_raw[9]})

ae52_raw.columns = ['datetime', 'ae52_BC', 'ae52_UVPM']

co2bg_raw = pd.DataFrame({'datetime' : co2bg_raw['datetime'],'co2bg_CO2ppm': co2bg_raw[3]})

co2dil_raw = pd.DataFrame({'datetime' : co2dil_raw['datetime'],'co2dil_CO2ppm': co2dil_raw[3]})

co2tpi_raw = pd.DataFrame({'datetime' : co2tpi_raw['datetime'],'co2tpi_CO2ppm': co2tpi_raw[3]})


drx_raw = pd.DataFrame({'datetime' : drx_raw.datetime,
                                'drxPM1' : drx_raw[2],
                                'drxPM2.5' : drx_raw[3],
                                'drxPM4' :drx_raw[4],
                                'drxPM10' : drx_raw[5],
                                'drxTPM' : drx_raw[6]})

fl_raw = pd.DataFrame({'datetime' : fl_raw['datetime'],
                    'flFP1_Flow' : fl_raw[1],
                    'flTemp1_F' : fl_raw[2],
                    'flPressure1_kPa' : fl_raw[3],
                    'flFP2_Flow' : fl_raw[4],
                    'flTemp2_F' : fl_raw[5],
                    'flPressure2_kPa' : fl_raw[6],
                    #'flFP3_Flow' : fl_raw[7],
                    #'flTemp3_F' : fl_raw[8],
                    #'flPressure3_kPa' : fl_raw[9],
                    'flVPO_Flow' : fl_raw[10],
                    'flTemp4_F' : fl_raw[11],
                    'flPressure4_kPa' : fl_raw[12],
                    'flDPI_Flow' : fl_raw[13],
                    'flTemp5_F' : fl_raw[14],
                    'flPressure5_kPa' : fl_raw[15]})


hnu_raw.columns = ['datetime','hnu']

testo_raw.columns = ['datetime','testoNO_ppm','testoO2_percent','testoCO2i_percent','testoCO_ppm','testoNO2_ppm','testoSO2_ppm',
                     'testoTf_C','testoTa_C','testoPump_Lmin','testoBatt_V','testodP_inW','testoPabs_hPa','testoCO2_percent','testoMCO2_kgh',
                     'testoPabs_hPa2']

lan_raw.columns = ['datetime','lannum','lanvolt','lantemp_C','lanco_ppm']


## interploation and wirting .csv files
vim_eng = fix(vim_eng_raw)
vim_eng.to_csv('vim_engine.csv')
et_eng = fix(et_eng_raw)
et_eng.to_csv('et_engine.csv')
lan = fix(lan_raw)
lan.to_csv('lan.csv')
ae52 = fix(ae52_raw)
ae52.to_csv('ae52.csv')
co2bg = fix(co2bg_raw)
co2bg.to_csv('co2bg.csv')
co2tpi = fix(co2tpi_raw)
co2tpi.to_csv('co2tpi.csv')
co2dil = fix(co2dil_raw)
co2dil.to_csv('co2dil.csv')
fl = fix(fl_raw)
fl.to_csv('flowrate.csv')
gps = fix(gps_raw)
gps.to_csv('gps.csv')
gps2 = fix(gps2_raw)
gps2.to_csv('gps2.csv')
hnu = fix(hnu_raw)
hnu.to_csv('hnu.csv')
drx = fix(drx_raw)
drx.to_csv('drx.csv')
testo = fix(testo_raw)

testo_CO2 = pd.DataFrame({'testo_CO2i' : testo['testoCO2i_percent'],'testo_CO2' : testo['testoCO2_percent']})
testo_CO2.to_csv('testo_co2.csv')

testo_CO = pd.DataFrame({'testo_COppm' : testo['testoCO_ppm']})
testo_CO.to_csv('testo_co.csv')

testo_etc = pd.DataFrame({
                    'testo_O2percent' : testo['testoO2_percent'],
                    'testo_SO2ppm' : testo['testoSO2_ppm'],
                    'testo_Ta_C' : testo['testoTa_C'],
                    'testo_Tf_C' : testo['testoTf_C'],
                    'testo_NOppm' : testo['testoNO_ppm'],
                    'testo_NO2ppm' : testo['testoNO2_ppm'],
                    'testo_Pump_Lmin' : testo['testoPump_Lmin'],
                    'testo_Pabs_hPa2' : testo['testoPabs_hPa2']})
testo_etc.to_csv('testo_etc.csv')

# merge all the TS together
dfs = [drx, lan, hnu, vim_eng, et_eng, ae52, co2bg, co2tpi, co2dil, fl, gps, gps2, testo_CO2, testo_CO, testo_etc]
#big = reduce(lambda left,right: pd.merge(left,right,on='datetime'), dfs)
big = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), dfs)
#big = big.set_index('datetime')
big.to_csv('big.csv')
# restrict window here:
#big = big[0:len(big)/3]

normed = pd.DataFrame({'drx' : normalize(big['drxPM2.5']),
                    'co2bg' : normalize(big.co2bg_CO2ppm),
                    'co2tpi' : normalize(big.co2tpi_CO2ppm),
                    'co2dil' : normalize(big.co2dil_CO2ppm),
                    'vim_rpm' : normalize(big.vim_eng_rpm),
		    'et_rpm' : normalize(big.et_eng_rpm),
                    'lan' : normalize(big.lanco_ppm),
                    'velocity' : normalize(big.gps_velocity),
                    #'velocity2' : normalize(big.gps2_velocity),
                    'ae52' : normalize(big.ae52_UVPM),
                    #'fl' : normalize(big.flVPO_Flow),
                    'hnu' : normalize(big.hnu),
                    'testo_co' : normalize(big.testo_COppm),
                    'testo_co2' : normalize(big.testo_CO2),
                    'testo_etc' : normalize(big.testo_SO2ppm)})
#normed = normed.set_index('datetime')
#normed.to_csv('normed.csv')

drx = lag(drx)
fl = lag(fl)
hnu = lag(hnu)

vim_eng_k = maxccf(normed.drx,normed.vim_rpm)
vim_eng_L = lag(vim_eng,vim_eng_k)
print 'drx lags vim_eng by', vim_eng_k, 'sec'

et_eng_k = maxccf(normed.drx,normed.et_rpm)
et_eng_L = lag(et_eng,et_eng_k)
print 'drx lags et_eng by', et_eng_k, 'sec'

lan_k = maxccf(normed.drx,normed.lan)
lan_L = lag(lan,lan_k)
print 'drx lags lan by', lan_k, 'sec'

ae52_k = maxccf(normed.drx,normed.ae52)
ae52_L = lag(ae52,ae52_k)
print 'drx lags ae52 by', ae52_k, 'sec'

co2bg_k = maxccf(normed.drx,normed.co2dil)
co2bg_L = lag(co2bg,co2bg_k)
print 'drx lags co2bg by', co2bg_k, 'sec'

co2tpi_k = maxccf(normed.drx,normed.co2tpi)
co2tpi_L = lag(co2tpi,co2tpi_k)
print 'drx lags co2tpi by', co2tpi_k, 'sec'

co2dil_k = maxccf(normed.drx,normed.co2dil)
co2dil_L = lag(co2dil,co2dil_k)
print 'drx lags co2dil by', co2dil_k, 'sec'

gps_k = maxccf(normed.drx,normed.velocity)
gps_L = lag(gps,gps_k)
print 'drx lags gps by', gps_k, 'sec'

gps2_k = gps_k #maxccf(normed.drx,normed.velocity2)
gps2_L = lag(gps2,gps_k) #using same lag as GPS
print 'drx lags gps2 by', gps_k, 'sec (using same lag as gps)'

testo_co2_k = -12#maxccf(normed.drx,normed.testo_co2)
testo_co2_L = lag(testo_CO2,testo_co2_k)
print 'drx lags testo_co2 by', testo_co2_k, 'sec'

testo_co_k = -12#maxccf(normed.drx,normed.testo_co)
testo_co_L = lag(testo_CO,testo_co_k) 
print 'drx lags testo_co by', testo_co_k, 'sec (using same lag as testo_co2)' 

testo_etc_k = -12#maxccf(normed.drx,normed.testo_etc)
testo_etc_L = lag(testo_etc,testo_etc_k)
print 'drx lags testo_etc by', testo_etc_k, 'sec'

f = open("lag.csv","w") #opens file with name of "test.txt"
f.write("ref, param, lag \n")
f.write("drx, vim_eng, %s\n" %vim_eng_k)
f.write("drx, et_eng, %s\n" %et_eng_k)
f.write("drx, lan, %s \n" %lan_k)
f.write("drx, ae52, %s \n" %ae52_k)
f.write("drx, co2bg, %s \n" %co2bg_k)
f.write("drx, co2tpi, %s \n" %co2tpi_k)
f.write("drx, co2dil, %s \n" %co2dil_k)
f.write("drx, gps, %s \n" %gps_k)
f.write("drx, gps2, %s\n" %gps_k)
f.write("drx, testo_co2, %s \n" %testo_co2_k)
f.write("drx, testo_co, %s \n" %testo_co_k)
f.write("drx, testo_etc, %s \n" %testo_etc_k)
f.close() 

dfsL = [drx, lan_L, hnu, vim_eng_L, et_eng_L, ae52_L, co2bg_L, co2tpi_L, co2dil_L, fl, gps_L, gps2_L, testo_co_L, testo_co2_L, testo_etc_L]
lagged = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'), dfsL)
## use drx.datetime to make a slice of this data.
lagged = pd.DataFrame(lagged, index=drx.index)
##
lagged = pd.DataFrame({
                'drxPM2.5' : lagged['drxPM2.5'],
                'drxPM1' : lagged['drxPM1'],
                'drxPM4' : lagged['drxPM4'],
                'drxTPM' : lagged['drxTPM'],
                'drxPM10' : lagged['drxPM10'],
                'lanco_ppm' : lagged.lanco_ppm,
                'lantemp_C' : lagged.lantemp_C,
                'gps_elevation' : lagged.gps_elevation,
                'gps_latitude' : lagged.gps_latitude,
                'gps_longitude' : lagged.gps_longitude,
                'gps_velocity' : lagged.gps_velocity,
                'gps2_elevation' : lagged.gps2_elevation,
                'gps2_latitude' : lagged.gps2_latitude,
                'gps2_longitude' : lagged.gps2_longitude,
                'gps2_velocity' : lagged.gps2_velocity,
                'ae52_UVPM' : lagged.ae52_UVPM,
                'ae52_BC' : lagged.ae52_BC,
                'zet_eng_rpm' : lagged.et_eng_rpm,
                'zet_eng_load' : lagged.et_eng_load,
                'zet_eng_fuel_rate' : lagged.et_eng_fuel_rate,
                'zet_eng_ground_spd' : lagged.et_eng_ground_spd,
                'vim_eng_ground_spd' : lagged.vim_eng_ground_spd,
                'vim_eng_payload' : lagged.vim_eng_payload,
                'vim_eng_rpm' : lagged.vim_eng_rpm,
                'vim_eng_load' : lagged.vim_eng_load,
                'vim_eng_fuel_rate' : lagged.vim_eng_fuel_rate,
                'co2bg_CO2ppm' : lagged.co2bg_CO2ppm,
                'co2tpi_CO2ppm' : lagged.co2tpi_CO2ppm,
                'co2dil_CO2ppm' : lagged.co2dil_CO2ppm,
                'flDPI_Flow' : lagged.flDPI_Flow,
                'flFP1_Flow' : lagged.flFP1_Flow,
                'flFP2_Flow' : lagged.flFP2_Flow,
                'flPressure1_kPa' : lagged.flPressure1_kPa,
                'flPressure2_kPa' : lagged.flPressure2_kPa,
                'flPressure4_kPa' : lagged.flPressure4_kPa,
                'flPressure5_kPa' : lagged.flPressure5_kPa,
                'flTemp1_F' : lagged.flTemp1_F,
                'flTemp2_F' : lagged.flTemp2_F,
                'flTemp4_F' : lagged.flTemp4_F,
                'flTemp5_F' : lagged.flTemp5_F,
                'flVPO_Flow' : lagged.flVPO_Flow,
                'hnu' : lagged.hnu,
                'testo_CO2i' : lagged.testo_CO2i,
                'testo_COppm' : lagged.testo_COppm,
                'testo_NOppm' : lagged.testo_NOppm,
                'testo_SO2ppm' : lagged.testo_SO2ppm,
                'testo_NO2ppm' : lagged.testo_NO2ppm,
		'testo_CO2' : lagged.testo_CO2,
		'testo_COppm' : lagged.testo_COppm,
		'testo_O2percent' : lagged.testo_O2percent,
		'testo_Ta_C' : lagged.testo_Ta_C,
		'testo_Tf_C' : lagged.testo_Tf_C,
		'testo_Pump_Lmin' : lagged.testo_Pump_Lmin,
		'testo_Pabs_hPa2' : lagged.testo_Pabs_hPa2})
##
lagged.to_csv('lagged.csv')
cor = pd.DataFrame({#'datetime' : normed.datetime,
                'drx_pm2.5' : lagged['drxPM2.5'],
                'vim_eng_fuel_rate' : lagged.vim_eng_fuel_rate,
		'et_eng_fuel_rate' : lagged.zet_eng_fuel_rate,
                'lan_COppm' : lagged.lanco_ppm,
                'ae52_uvpm' : lagged.ae52_UVPM,
                'ae52_bc' : lagged.ae52_BC,
                #'eng_hp' : lagged.eng_hp,
                'et_eng_rpm' : lagged.zet_eng_rpm,
                'et_eng_load' : lagged.zet_eng_load,
                'vim_eng_rpm' : lagged.vim_eng_rpm,
                'vim_eng_load' : lagged.vim_eng_load,
                'co2bg' : lagged.co2bg_CO2ppm,
                'co2tpi' : lagged.co2tpi_CO2ppm,
                'co2dil' : lagged.co2dil_CO2ppm,
                #'fl' : normed.fl,
                'velocity' : lagged.gps_velocity,
                'hnu' : lagged.hnu,
                'testo_CO2i' : lagged.testo_CO2i,
                'testo_COppm' : lagged.testo_COppm,
                'testo_NOppm' : lagged.testo_NOppm,
                'testo_SO2ppm' : lagged.testo_SO2ppm,
                'testo_NO2ppm' : lagged.testo_NO2ppm})
#df = df.set_index('datetime')
#cor.to_csv('compared.csv')
cor = cor.dropna()
cor.corr().to_csv("cor_matrix.csv")
###
old = pd.DataFrame(big, index=lagged.index)
## find a way to plot this with out having to normalize
t = lagged.index
y1 = normalize(lagged.drxTPM)
y2 = normalize(old.testo_CO2i)
y3 = normalize(lagged.testo_CO2i)
fig = plt.figure()
#axis, first plot
ax1 = fig.add_subplot(211)
ax1.plot(t, y1)
ax1.plot(t, y2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)
## other plot
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, y1)
ax2.plot(t, y3)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)
#plt.show()
