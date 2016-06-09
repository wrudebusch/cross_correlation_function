import pandas as pd
import matplotlib.pyplot as plt

def normalize(A):
    A -= A.mean(); A /= A.std()
    return A

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') 

lagged = pd.read_csv('lagged.csv', parse_dates={'datetime': [0]}, date_parser=dateparse)
lagged = lagged.set_index('datetime')

big = pd.read_csv('big.csv', parse_dates={'datetime': [0]}, date_parser=dateparse)
big = big.set_index('datetime')
old = big.reindex(lagged.index)

## find a way to plot this with out having to normalize
t = lagged.index
y1 = normalize(lagged.drxTPM)
y2 = normalize(old.testo_NOppm)
y3 = normalize(lagged.testo_NOppm)
fig = plt.figure()
#axis, first plot
ax1 = fig.add_subplot(211)
ax1.plot(t, y1)
ax1.plot(t, y2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)
plt.ylabel('original testo_NOppm vs. DRX')
## other plot
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, y1)
ax2.plot(t, y3)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)
plt.ylabel('lagged testo_NOppm')
plt.title('Cross Correlation Function; Before and After')
#plt.show()

### 2nd graph
step = len(lagged)/4
lagged_step = lagged[3*step:-1]
old_step = old[3*step:-1]
t = lagged_step.index
y1 = normalize(lagged_step.drxTPM)
y2 = normalize(old_step.testo_NOppm)
y3 = normalize(lagged_step.testo_NOppm)
fig = plt.figure()
#axis, first plot
ax1 = fig.add_subplot(211)
ax1.plot(t, y1)
ax1.plot(t, y2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)
plt.ylabel('original testo_NOppm vs. DRX')
## other plot
plt.title('Before Cross Correlation Function')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, y1)
ax2.plot(t, y3)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)
plt.ylabel('lagged testo_NOppm')
plt.title('After Cross Correlation Function')
plt.show()

### 3rd graph
t = lagged.index
y1 = normalize(lagged.drxTPM)
y2 = normalize(old.vim_eng_rpm)
y3 = normalize(lagged.vim_eng_rpm)
fig = plt.figure()
#axis, first plot
ax1 = fig.add_subplot(211)
ax1.plot(t, y1)
ax1.plot(t, y2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)
plt.ylabel('original vim_eng_rpm vs. DRX')
## other plot
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, y1)
ax2.plot(t, y3)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)
plt.ylabel('lagged vim_eng_rpm')
#plt.title('Histogram of IQ')
plt.title('Cross Correlation Function; Before and After')
plt.show()
