#!/usr/bin/env python
# coding: utf-8

# # Project 1: Processing health and activity data [40 marks]
# 
# ---
# 
# Make sure you read the instructions in `README.md` before starting! In particular, make sure your code is well-commented, with sensible structure, and easy to read throughout your notebook.
# 
# ---
# 
# The MMASH dataset [1, 2] is is a dataset of health- and activity-related measurements taken on 22 different people, over a continuous period of 24 hours, using wearable devices.
# 
# In this project, we have provided you with some of this data for **10** of those individuals. In the `dataset` folder, you will find:
# 
# - a file `subject_info.txt` which summarises the age (in years), height (in cm), and weight (in kg) of all 10 study participants,
# - 10 folders named `subject_X`, which each contain two files:
#     - `heartbeats.txt` contains data on all individual heartbeats detected over the 24-hour study period,
#     - `actigraph.txt` contains heart rate and other activity data measured with another device, over the same 24-hour period.
# 
# The tasks below will guide you through using your Python skills to process some of this data. Note that the data was reformatted slightly for the purpose of the assignment (to make your life a bit easier!), but the values are all the original ones from the real dataset.
# 
# ### Getting stuck
# 
# Tasks 3 to 8 follow directly from each other. There is a `testing` folder provided for you with `.npy` files and a supplementary `actigraph.txt` dataset. The `.npy` files are NumPy arrays, which you can load directly using `np.load()`, containing an example of what the data should look like after each task. You will be able to use this example data to keep working on the later tasks, even if you get stuck on an earlier task. Look out for the 💾 instructions under each task.
# 
# These were produced using the data for another person which is not part of the 10 you have in your dataset.
# 
# 
# ### References
# 
# [1] Rossi, A., Da Pozzo, E., Menicagli, D., Tremolanti, C., Priami, C., Sirbu, A., Clifton, D., Martini, C., & Morelli, D. (2020). Multilevel Monitoring of Activity and Sleep in Healthy People (version 1.0.0). PhysioNet. https://doi.org/10.13026/cerq-fc86
# 
# [2] Rossi, A., Da Pozzo, E., Menicagli, D., Tremolanti, C., Priami, C., Sirbu, A., Clifton, D., Martini, C., & Morelli, D. (2020). A Public Dataset of 24-h Multi-Levels Psycho-Physiological Responses in Young Healthy Adults. Data, 5(4), 91. https://doi.org/10.3390/data5040091.
# 
# ---
# ## Task 1: Reading the subject information
# 
# The file `subject_info.txt` in your `dataset` folder summarises the age (in years), height (in cm), and weight (in kg) of all 10 study participants.
# 
# ---
# 🚩 ***Task 1:*** Write a function `read_subject_info()` which reads in the information in `subject_info.txt`, and returns two outputs:
# 
# - a list `headers` containing the four column headers as strings, read from the first line in the file;
# - a NumPy array `info` containing the numerical information for each person (i.e. it should have 10 rows and 4 columns).
# 
# **Important:** the height of each subject should be given in **metres** in your `info` array.
# 
# **[3 marks]**

# In[2]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import interp1d
import math
import numpy as np
def read_subject_info(dir):
    with open(dir, 'r') as f:       # open the txt file
        for num, line in enumerate(f):    #  combine a traversable data object, such as a list, tuple, or string, into an index sequence that lists both the data and the data subscript
            if num == 0:
                str = line.strip('\n')    #delete the Line breaks
                break

    header = str.split(',') 

    info = np.loadtxt(dir, skiprows=1, delimiter=',') 
    info[:, 2] = info[:, 2] * 0.01 # Conversion of units

    return header, info
filepath = 'dataset\\subject_info.txt'
header, info = read_subject_info(filepath)
print(header)
print(info)


# ---
# ## Task 2: Charting the Body Mass Index (BMI) for all participants
# 
# The Body Mass Index (BMI) can be used to indicate whether someone is at a healthy body weight. [The NHS website](https://www.nhs.uk/common-health-questions/lifestyle/what-is-the-body-mass-index-bmi/) describes it as follows:
# 
# > The body mass index (BMI) is a measure that uses your height and weight to work out if your weight is healthy.
# >
# > The BMI calculation divides an adult's weight in kilograms by their height in metres, squared. For example, a BMI of $25$ means $25 \text{kg/m}^2$.
# >
# > For most adults, an ideal BMI is in the $18.5$ to $24.9$ range.
# 
# This means that the BMI is calculated as follows:
# 
# $$
# \text{BMI} = \frac{\text{weight}}{\text{height}^2}.
# $$
# 
# ---
# 🚩 ***Task 2:*** Write a function `bmi_chart(info)` which takes as input the `info` array returned by `read_subject_info()`, produces a visualisation showing all subjects' heights and weights on a graph, and clearly indicates whether they are within the "healthy weight" range as described above (i.e. their BMI is in the $18.5$ to $24.9$ range).
# 
# Your function should not return anything, but calling it with `bmi_chart(info)` must be sufficient to display the visualisation.
# 
# You should choose carefully how to lay out your plot so that it is easy to interpret and understand.
# 
# **[4 marks]**

# In[3]:


def bmi_chart(info):
    people = info[:, 0]#  all the first column data
    weight = info[:, 1]#  all the second column data
    height = info[:, 2]#  all the tired column data

    bmi = weight / height ** 2

    plt.scatter(people, bmi, label='BMI')
    plt.axhline(y=18.5, color='r', linestyle='-') #the line for bmi = 18.9
    plt.axhline(y=24.9, color='r', linestyle='-') #the line for bmi = 24.9
    plt.text(0.6, 29, '95kg,1.83m', fontsize=12)
    plt.text(1.2, 21.4, '80kg,1.96m', fontsize=12)  #create the data
    plt.text(2.2, 20.1, '62kg,1.78m', fontsize=12)  #creare the data
    plt.text(4.2, 19.5, '65kg,1.83m', fontsize=12)  #create the data
    plt.text(3.9, 22.4, '74kg,1.84m', fontsize=12)  #create the data
    plt.text(5.3, 23.4, '70kg,1.75m', fontsize=12)  #create the data
    plt.text(5.9, 32.2, '115kg,1.86m', fontsize=12)  #create the data
    plt.text(7.2, 25.3, '80kg,1.80m', fontsize=12)  #create the data
    plt.text(8.2, 23.4, '70kg,1.75m', fontsize=12)  #create the data
    plt.text(9.2, 20.8, '92kg,2.05m', fontsize=12)  #create the data
    plt.text(9.5, 25, 'bmi=24.9', color='r', fontsize=14)
    plt.text(9.5, 18.6, 'bmi=18.5', color='r', fontsize=14)
    plt.xticks(np.arange(1, 11, 1))
    plt.title('The Weight, Height and BMI of every student', fontsize=15)
    plt.xlabel('Subject', fontsize=13)
    plt.ylabel('BMI($\mathrm{kg/m^2}$)', fontsize=13) # create the y label
    plt.legend(loc='upper right', fontsize=14)       #create the legend and put it in the upper right position
    plt.show()
             
bmi_chart(info)


# ---
# ## Task 3: Instantaneous heart rate data
# 
# For each subject, the file `heartbeats.txt` contains data on all individual heartbeats detected over the 24-hour study period. Specifically, the two columns record the time at which each heartbeat was detected, and the interval (in seconds) between the current heartbeat and the previous one.
# 
# ### Handling timestamp data
# 
# For the next tasks, you will use NumPy's `datetime64[s]` and `timedelta64[s]` object types, respectively used to represent times (as if read on a clock) and time intervals. You should [consult the relevant documentation](https://numpy.org/doc/stable/reference/arrays.datetime.html#datetimes-and-timedeltas).
# 
# Here are a few illustrative examples:

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Create two datetime objects and a vector of dates
start_time = np.datetime64('2022-10-20 12:00:00')
end_time = np.datetime64('2022-11-10 12:00:00')
time_vector = np.array(['2022-10-20', '2022-10-23', '2022-10-28'], dtype='datetime64[s]')
print(time_vector)

# Get time interval between the two times
time_elapsed = end_time - start_time
print(time_elapsed)
print(type(time_elapsed))

# Divide by the duration of 1 second to get number of seconds (as a number object)
seconds_elapsed = time_elapsed / np.timedelta64(1, 's')
print(seconds_elapsed)
print(type(time_elapsed))

# Divide by the duration of 1 day to get number of days
days_elapsed = time_elapsed / np.timedelta64(1, 'D')
print(days_elapsed)

# Create a range of datetimes spaced by 1 day
step = np.timedelta64(1, 'D')
days = np.arange(start_time, end_time + step, step)#

# Plot something using days as the x-axis
fig, ax = plt.subplots(figsize=(12, 4))
value = np.random.randint(1, 11, size=len(days))
ax.plot(days, value, 'ro-')
ax.set(ylim=[0, 11], xlabel='Date', ylabel='Value')
plt.show()


# ---
# 🚩 ***Task 3a:*** Write a function `read_heartbeat_data(subject)` which takes as input an integer `subject` between 1 and 10, reads the data in `heartbeats.txt` for the given `subject`, and returns it as two NumPy vectors:
# 
# - `times`, containing the recorded times of each heartbeat (as `datetime64[s]` objects),
# - `intervals`, containing the recorded intervals between heartbeats (in seconds, as `float` numbers).
# 
# **[3 marks]**

# In[4]:


def read_heartbeat_data(subject):
    path = 'dataset\\subject_' + str(subject) + '\\heartbeats.txt'     #get the subject_x data

    intervals = np.loadtxt(path, skiprows=1, delimiter=',', usecols=2) #get the interval data

    with open(path) as f:
        read_data = f.read()
        a = read_data.split(',')
    f.closed

    times = np.array(a[3::2], dtype='datetime64')  #get the times data

    return times, intervals
times, intervals = read_heartbeat_data(2)
print(times)
print(intervals)
    


# ---
# 🚩 ***Task 3b:*** Write a function `hr_from_intervals(intervals)` which takes as input a NumPy vector containing heartbeat interval data (such as that returned by `read_heartbeat_data()`), and returns a NumPy vector of the same length, containing the instantaneous heart rates, in **beats per minute (BPM)**, calculated from the intervals between heartbeats. You should return the heart rates as floating-point numbers.
# 
# For instance, an interval of 1 second between heartbeats should correspond to a heart rate of 60 BPM.
# 
# **[2 marks]**

# In[5]:


def hr_from_intervals(intervals):
    BPM = 60 / intervals  #BPM means beats per minutes

    return BPM

BPM = hr_from_intervals(intervals)
print(BPM)


# ---
# ## Task 4: Data cleaning
# 
# There are gaps and measurement errors in the heartbeat data provided by the device. These errors will likely appear as outliers in the data, which we will now try to remove.
# 
# One possible method is to remove data points which correspond to values above and below certain **percentiles** of the data. Removing the data below the $p$th percentile means removing the $p\%$ lowest values in the dataset. (Note that, for instance, the 50th percentile is the median.)
# 
# ---
# 🚩 ***Task 4a:*** Write a function `clean_data(times_raw, hr_raw, prc_low, prc_high)` which takes 4 inputs:
# 
# - `times_raw` is the NumPy array of timestamps returned by `read_heartbeat_data()`,
# - `hr_raw` is the NumPy array of computed heart rate values returned by `hr_from_intervals()`,
# - `prc_low` and `prc_high` are two numbers such that $0\leq$ `prc_low` $<$ `prc_high` $\leq 100$.
# 
# Your function should return two NumPy arrays of the same length, `times` and `hr`, which are the original arrays `times_raw` and `hr_raw` where all the measurements (heart rate and associated time stamp) below the `prc_low`th percentile and above the `prc_high`th percentile of the heart rate data have been removed.
# 
# You may wish to make use of NumPy functionality to calculate percentiles.
# 
# **[4 marks]**

# In[6]:


def clean_data(times_raw, hr_raw, prc_low, prc_high):
    index = np.argsort(hr_raw) #After sorting the elements in the hr_law from smallest to largest, extract the corresponding index

    low_b = int(prc_low * len(index) / 100)           #calculate the percentage number below prc_low
    high_b = math.ceil(prc_high * len(index) / 100)   # rounded up to the nearest integer，alculate the percentage number over prc_low

    index_sort = index[low_b:high_b] #filter data
    sort = np.sort(index_sort) # Reorder it once for later task 5 

    hr_raw_sort = hr_raw[sort]#filter hr data
    times_raw_sort = times_raw[sort]#filer time dara

    return times_raw_sort, hr_raw_sort

times_sort, hr_sort = clean_data(times, BPM, 1, 99)
print(times_sort)
print(hr_sort)


# ---
# 🚩 ***Task 4b:*** Write a function `evaluate_cleaning(subject)`, which takes as input an integer `subject` between 1 and 10 indicating the subject number, and plots the following two histograms for that subject:
# 
# - a histogram of the raw heart rate data,
# - a histogram of the heart rate data after cleaning with `clean_data()`, where the bottom 1% and the top 1% of the values have been removed.
# 
# Your histograms should use a logarithmic scale on the y-axis, and be clearly labelled. You should consider carefully how to lay out the histogram to best present the information.
# 
# Your function `evaluate_cleaning()` should call the functions `read_heartbeat_data()`, `hr_from_intervals()`, and `clean_data()` you wrote above, in order to obtain the raw and cleaned heart rate data for a given `subject`.
# 
# Then, use your function to display the histograms of the raw and cleaned data for Subject 3. Given that heart rates in adults can typically range from about 40 to 160 beats per minute, and given your histograms, explain why this is a suitable method to remove likely measurement errors in the heart rate data.
# 
# **[3 marks]**
# 
# ---
# 
# 💾 *If you are stuck on Task 3 or on the task above, you can load the data provided in the `testing` folder to produce your histograms, by running the following commands:*
# 
# ```python
# times_raw = np.load('testing/times_raw.npy')
# hr_raw = np.load('testing/hr_raw.npy')
# times = np.load('testing/times.npy')
# hr = np.load('testing/hr.npy')
# ```

# In[7]:


def evaluate_cleaning(subject):
    time, intervals = read_heartbeat_data(subject) #get the time and intervals data
    BPM_b = hr_from_intervals(intervals)           #get the BPM result
    times_a, hr_a = clean_data(time, BPM_b, 1, 99)

    plt.yscale('log')
    plt.hist(BPM_b, bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], facecolor="blue",
             edgecolor="black", alpha=0.3, label='BPM before cleaning')
    plt.hist(hr_a, bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], facecolor="red",
             edgecolor="black", alpha=0.3, label='BPM after cleaning')
    plt.ylim(1e1, 1e5)
    plt.xticks(np.arange(0, 250, 20)) #  get x label
    plt.xlabel('Beats Per Minute', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.title('Comparison between BMP Before and After Cleaning', fontsize=12)
    plt.show()
evaluate_cleaning(3)


# *Because the extreme values are filtered by the conditions of the upper and lower percentage values, the measurement of the data is more accurate.*

# #---
# ## Task 5: Interpolating the data
# 
# Although the device detecting heartbeats was able to measure intervals between beats with millisecond precision, the recorded timestamps could only record the second at which a heartbeat occurred. This means that there are not only time gaps in the data (due to the device missing heartbeats), but also several heartbeats usually recorded in the same second.
# 
# For example, this is an excerpt from Subject 7's data, showing a 9-second time gap between `09:19:57` and `09:20:06`, as well as 3 different heartbeats detected at `09:20:06`:
# 
# ```
# 59,2022-07-21 09:19:56,1.033
# 60,2022-07-21 09:19:57,0.942
# 61,2022-07-21 09:20:06,0.307
# 62,2022-07-21 09:20:06,0.439
# 63,2022-07-21 09:20:06,0.297
# 64,2022-07-21 09:20:07,0.427
# ```
# 
# The goal of this next task is to **interpolate** the recorded data, in order to produce a new dataset containing values of the heart rate at regular time intervals. We will use **linear interpolation**, with the help of SciPy's `interp1d()` function (from the `interpolate` module) which we saw in Week 5.
# 
# ---
# 🚩 ***Task 5a:*** The `interp1d()` function from SciPy can only be used with numeric data, and not timestamps. Two functions are provided for you below.
# 
# - Explain, in your own words, what both functions do and how.
# - Write a few lines of test code which clearly demonstrate how the functions work.
# 
# **[2 marks]**

# In[8]:


def datetime_to_seconds(times):
    return (times - times[0]) / np.timedelta64(1, 's')
    #Returns the interval between two datetime type time
def seconds_to_datetime(seconds_elapsed, start_time):
    return seconds_elapsed * np.timedelta64(1, 's') + start_time
    #Returns the datetime after the given number of seconds

#demonstration of use
times, intervals = read_heartbeat_data(2)
tmp1  = datetime_to_seconds(times)
print(tmp1)

tmp2 = seconds_to_datetime(tmp1, times[0])
print(tmp2)


# *the function `datetime_to_seconds()`  is give a times array,it will return the element represents the time interval of time(seconds)
# the function `second_to_datetime()`  is return the exact datetime when knowing the given interval of time(seconds) and the based staring time*

# ---
# 🚩 ***Task 5b:*** Write a function `generate_interpolated_hr(times, hr, time_delta)` which takes as inputs:
# 
# - two NumPy vectors `times` and `hr` such as those returned by `clean_data()`,
# - a `timedelta64[s]` object representing a time interval in seconds,
# 
# and returns two new NumPy vectors, `times_interp` and `hr_interp`, such that:
# 
# - `times_interp` contains regularly spaced `datetime64[s]` timestamps, starting at `times[0]`, ending on or less than `time_delta` seconds before `times[-1]`, and with an interval of `time_delta` between consecutive times.
# - `hr_interp` contains the heart rate data obtained using **linear interpolation** and evaluated at each time in `times_interp`, for example with the help of the `interp1d()` function from `scipy.interpolate`.
# 
# For example, if `times` starts at `10:20:00` and ends at `10:20:09` with a `time_delta` of two seconds, then your `times_interp` vector should contain `10:20:00`, `10:20:02`, `10:20:04`, `10:20:06`, `10:20:08`, and `hr_interp` should consist of the corresponding interpolated heart rate values at each of those times.
# 
# **[4 marks]**

# In[9]:


def generate_interpolated_hr(times, hr, time_delta):
    times_second = datetime_to_seconds(times) #Get the irregular time interval duration in the dataset
    length = int(times_second[-1] / time_delta + 1)# get the number of whole x point which will be populated the data 
    times_second_interp = np.zeros(length)
    
    hr_interp = np.zeros(length)
    for i in range(length):
        times_second_interp[i] = time_delta * i# Set the linear  X value 

    f = interp1d(times_second, hr, kind='linear', bounds_error=False, fill_value=0)# Build a linear interpolation model

    for i in range(length):
        hr_interp[i] = f(times_second_interp[i])#Prediction of seconds Y based on the new x-axis

    times_interp = seconds_to_datetime(times_second_interp, times[0])#transfer "second" data into datetime

    return times_interp, hr_interp

    


# ---
# 🚩 ***Task 5c:*** Check your `generate_interpolated_hr()` function by generating interpolated heart rate data for Subject 1 for just the first 100 measurements (after cleaning). You should generate interpolated data with a time interval of 5 seconds. Plot the data points, as well as your interpolated data, and discuss in the Markdown cell below whether your plot is what you expected, and why.
# 
# **[2 marks]**
# 
# ---
# 💾 *If you are stuck on Task 4, you can use the cleaned data provided in the `testing` folder to check your code, by running the following commands:*
# 
# ```python
# times = np.load('testing/times.npy')
# hr = np.load('testing/hr.npy')
# ```

# In[10]:


times_interp, hr_interp = generate_interpolated_hr(times_sort[:100], hr_sort[:100], 5) #with a interval o 5 seconds for first 100 measurements
plt.plot(times_sort[:100], hr_sort[:100])
plt.plot(times_interp, hr_interp)
plt.show()


# *The result is what i am expected, in the time of 5 second：from 14：09：25 to 14:09:29 ,i interpolate data in a linear way,just a plot shows.*

# ---
# ## Task 6: Smoothing the data with a rolling average
# 
# A rolling average is simply an average of the heart rate data, calculated over a given window of time. For example:
# 
# - The 20-second rolling average of the heart rate at a time `10:20:00` is the average heart rate over the 20 seconds leading up to that time, i.e. the average of all the heart rates between `10:19:41` and `10:20:00` (inclusive). If we have measurements of the heart rate every 5 seconds, then this would be the average of the heart rates measured at `10:19:45`, `10:19:50`, `10:19:55`, and `10:20:00`.
# - We can similarly calculate the 20-second rolling average at the next measurement time, `10:20:05`, as the average heart rate over the 20-second period from `10:19:46` to `10:20:05` (inclusive).
# 
# The rolling average essentially smoothes out the sudden jumps in the measured (or interpolated) heart rate data, allowing us to see the longer-term variations more clearly.
# 
# ---
# 🚩 ***Task 6:*** Write a function `rolling_average()` which takes as inputs:
# 
# - two NumPy vectors `times` and `hr` such as those returned by `clean_data()`,
# - a `timedelta64[s]` object `time_delta` representing a time interval in seconds,
# - a `timedelta64[s]` object `window`, representing the window duration in seconds (with `window` assumed to be an integer multiple of `time_delta`),
# 
# and returns a NumPy vector `hr_rolling` containing values for the rolling average of the heart rate over time, with the given window size.
# 
# Your `rolling_average()` function should call `generate_interpolated_hr()` to generate regularly-spaced heart rate data with a time interval `time_delta`, before computing and returning the averaged heart rate data.
# 
# Note that `hr_rolling` will be shorter than the length of your interpolated heart rate data, because you can only start computing rolling averages after one window of time has elapsed. (For instance, if your data starts at `10:20:00`, with a 30-second window, the first value of the rolling average you can obtain is at `10:20:29`.)
# 
# **[4 marks]**

# In[11]:


def rolling_average(times, hr, time_delta, window):

    times_interp, hr_interp = generate_interpolated_hr(times, hr, time_delta)

    hr_rolling = np.zeros(len(times_interp) - window) # when rolling like a window, we should know the step number of rolling and make a corrsponding numpy zero array

    for i in range(len(hr_rolling)):

        hr_rolling[i] = np.sum(hr_interp[i:window + i]) / window #calculate the average result after one window of time

    return times_interp[:-window],hr_rolling #the task only ask to return hr_rolling,but for the convenience of later task7,I return the times_interp[:-window] as well
times_rolling,hr_rolling = rolling_average(times_sort, hr_sort, 5, 20) #test the result
print(hr_rolling)


# ---
# ## Task 7: Putting it all together
# 
# You should now have a series of functions which allow you to:
# 
# - read data on measured heartbeart-to-heartbeat intervals for a given subject,
# - transform this data into heart rate measurements and clean out the outliers,
# - interpolate the data to generate measurements at regular time intervals,
# - compute a rolling average of the heart rate data over time, to smooth out the data.
# 
# For each subject, there is another file `actigraph.txt`, containing activity data recorded by a separate device. In particular, this data provides another independent measurement of the subjects' heart rate. We can use this to check our work.
# 
# ---
# 🚩 ***Task 7:*** Write a function `display_heart_rate(subject)` which takes as input an integer `subject` between 1 and 10, and produces one single graph, containing two plots on the same set of axes:
# 
# - a plot of the heart rate data found in `actigraph.txt` over time,
# - a plot of the smoothed heart rate data computed by you from the data in `heartbeats.txt`, using interpolated measurements of the heart rate every 3 seconds, and a 30-second window size for the averaging.
# 
# Your plot should show good agreement between the two sets of data. Instead of showing the full 24 hours of data, you should choose a period of time over which to plot the heart rate (say, approximately 1 hour), in order to better visualise the results.
# 
# Show an example by using your function to display the results for 3 subjects of your choice.
# 
# **[4 marks]**
# 
# ---
# 💾 *If you are stuck on Task 5 or 6, you can use the actigraph heart rate data provided in the `testing` folder in `actigraph.txt`, and compare this to the smoothed heart rate data provided in the `testing` folder, which you can load by running the following command:*
# 
# ```python
# hr_rolling = np.load('testing/hr_rolling.npy')
# ```

# In[12]:


def display_heart_rate(subject):

    path = 'dataset\\subject_' + str(subject) + '\\actigraph.txt' # choose subject_x's actigraph.txt data

    hr = np.loadtxt(path, skiprows=1, delimiter=',', usecols=2)   #get the heart rate data
    with open(path) as f:
        read_data = f.read()
        a = read_data.split(',')
    f.closed
    times = np.array(a[8::7], dtype='datetime64')           

    times_h, intervals_h = read_heartbeat_data(subject)  #get the times and intervals data
    BPM_h = hr_from_intervals(intervals_h)               #get the BPM data
    times_sort, hr_sort = clean_data(times_h, BPM_h, 1, 99) #clean the data that the bottom 1% and the top 1% of the values have been removed.
    times_rolling, hr_rolling = rolling_average(times_sort, hr_sort, 3, 30) #using interpolated measurements of the heart rate every 3 seconds, and a 30-second window size for the averaging
    
    plt.plot(times, hr)
    plt.plot(times_rolling, hr_rolling)
    plt.show()
display_heart_rate(1) #result of subject_1
display_heart_rate(2) #result of subject_2
display_heart_rate(3) #result of subject_3


# ---
# ## Task 8: relating to other data
# 
# The data in `actigraph.txt` also contains the following columns:
# 
# - `Steps` indicates the number of steps detected per second (using a pedometer).
# - `Inclinometer Standing`/`Sitting`/`Lying` indicates the position of the subject, automatically detected by the device.
# - `Inclinometer Off` indicates when the device didn't record a position.
# 
# In particular, the `Inclinometer ...` columns record either `0` or `1`, and they are mutually exclusive over each row. This means that, for example, a subject can't be recorded simultaneously sitting and standing.
# 
# ---
# 🚩 ***Task 8:*** Using the results of your data processing work in previous tasks, can you relate some of this additional data (and/or some of the data in `subject_info.txt`) to the heart rate estimates that you have obtained?
# 
# You are free to choose how you complete this task. You will be assessed on the correctness of your code and analysis, the quality of your code (readability, commenting/documentation, structure), and the presentation of your results.
# 
# Note that you do not have to use **all** of the extra data to obtain full marks.
# 
# **[5 marks]**
# 
# ---
# 💾 *If you are using `hr_rolling.npy` and the actigraph data in the `testing` folder, this is the information for this person:*
# 
# | Weight | Height | Age |
# |:-:|:-:|:-:|
# | 85 | 180 | 27 |

# In[ ]:




