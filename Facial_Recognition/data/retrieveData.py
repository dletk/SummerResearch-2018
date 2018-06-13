# %% Libraries
import requests

# %% Load all the link from facescrub database
# Read the input file line by line
fileInput = open("./facescrub_actors.txt", "rb")
# Read all the lines in the file and store into a list
# The first line of the file is headers, so skip it
rawData = fileInput.readlines()[1:]
fileInput.close()

# Check the data
print(len(rawData))

# %% Convert the data to string instead of bytes, and make it more readable
data = []
labels = []
for line in rawData:
    # Split the line into a list, separated by white space
    line_data = line.split()
    # Get the name of the actor/actress from the data and the link to the image url
    name = line_data[0].decode("utf-8") + " " + line_data[1].decode("utf-8")
    url = line_data[-3].decode("utf-8")
    # Save the extracted data to new lists
    data.append(url)
    labels.append(name)

# %% Check the processed data
print(data[11234])
print(labels[11234])

# %% Make a request for each image url and save it as an image to the database, with list of label attached
for i in range(10)
    response = requests.get(data[i])
    if response.status_code == 200:
        with open("./faceScrub/"+str(i)+".jpg", "wb") as fileImage:
            fileImage.write(response.content)
            
