import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

# Define columns for the final output
cols = ['age', 'gender', 'path', 'name', 'dob', 'photo_taken', 'face_location', 'face_score1', 'face_score2', 'celeb_id']

# Define mat file paths
imdb_mat = 'imdb_crop/imdb.mat'
wiki_mat = 'wiki_crop/wiki.mat'

# Load .mat files
print("Loading IMDB and WIKI datasets...")
imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

# Free up memory
del imdb_mat, wiki_mat

# Extract dataset structures
imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

# Print structure information for debugging
print("IMDB data structure:", imdb.shape)
print("IMDB[0][0] fields count:", len(imdb[0][0]))

# Extract IMDB fields
print("Extracting IMDB fields...")
imdb_dob = imdb[0][0][0][0]  # date of birth (Matlab serial date number)
imdb_photo_taken = imdb[0][0][1][0]  # year when the photo was taken
imdb_full_path = imdb[0][0][2][0]  # path to file
imdb_gender = imdb[0][0][3][0]  # 0 for female and 1 for male, NaN if unknown

# Safely extract remaining fields with error checking
try:
    imdb_name = imdb[0][0][4][0]  # name of the celebrity
    print(f"Successfully extracted imdb_name, length: {len(imdb_name)}")
except Exception as e:
    print(f"Error extracting imdb_name: {e}")
    imdb_name = np.array([["unknown"]] * len(imdb_dob))

try:
    imdb_face_location = imdb[0][0][5][0]  # location of the face
    print(f"Successfully extracted imdb_face_location, length: {len(imdb_face_location)}")
except Exception as e:
    print(f"Error extracting imdb_face_location: {e}")
    imdb_face_location = np.array([[0, 0, 0, 0]] * len(imdb_dob))

imdb_face_score1 = imdb[0][0][6][0]  # detector score
imdb_face_score2 = imdb[0][0][7][0]  # second face detector score

# Safely extract celebrity names and IDs
try:
    imdb_celeb_names = imdb[0][0][8][0]  # list of all celebrity names
    print(f"Successfully extracted imdb_celeb_names, length: {len(imdb_celeb_names)}")
except Exception as e:
    print(f"Error extracting imdb_celeb_names: {e}")
    imdb_celeb_names = np.array([["unknown"]])

try:
    imdb_celeb_id = imdb[0][0][9][0]  # index of celebrity name
    print(f"Successfully extracted imdb_celeb_id, length: {len(imdb_celeb_id)}")
    print(f"Sample imdb_celeb_id types: {type(imdb_celeb_id[0])}")
except Exception as e:
    print(f"Error extracting imdb_celeb_id: {e}")
    imdb_celeb_id = np.array([0] * len(imdb_dob))

# Extract WIKI fields
print("Extracting WIKI fields...")
wiki_dob = wiki[0][0][0][0]
wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_location = wiki[0][0][5][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

# Process paths
imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb_crop/' + path[0])

for path in wiki_full_path:
    wiki_path.append('wiki_crop/' + path[0])

# Process gender
imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

# Process celebrity names
print("Processing celebrity names...")
imdb_names = []
for name in imdb_name:
    try:
        if isinstance(name, np.ndarray) and name.size > 0:
            if isinstance(name[0], np.ndarray):
                # Handle case where name is an array of characters
                try:
                    imdb_names.append(''.join(char[0] for char in name[0]))
                except:
                    imdb_names.append('unknown')
            else:
                # Handle case where name[0] is not an array
                try:
                    imdb_names.append(str(name[0]))
                except:
                    imdb_names.append('unknown')
        else:
            # Handle case where name is a scalar
            try:
                imdb_names.append(str(name))
            except:
                imdb_names.append('unknown')
    except Exception as e:
        print(f"Error processing name: {e}, type: {type(name)}")
        imdb_names.append('unknown')

print(f"Processed {len(imdb_names)} celebrity names")

# Process celebrity IDs
print("Processing celebrity IDs...")
imdb_celeb_indices = []
for i in range(len(imdb_celeb_id)):
    try:
        if np.isscalar(imdb_celeb_id[i]):
            # Handle scalar case
            val = imdb_celeb_id[i]
            imdb_celeb_indices.append(int(val) if not np.isnan(val) else -1)
        else:
            # Handle array case
            imdb_celeb_indices.append(-1)
    except Exception as e:
        print(f"Error processing celeb_id at index {i}: {e}")
        imdb_celeb_indices.append(-1)

# Process face locations
print("Processing face locations...")
imdb_face_locs = []
wiki_face_locs = []

for loc in imdb_face_location:
    try:
        # Simplified approach - just extract integers or use zeros
        if isinstance(loc, np.ndarray) and loc.size >= 4:
            coords = ["0", "0", "0", "0"]
            for i in range(4):
                try:
                    # Try different approaches to extract the value
                    if np.isscalar(loc[i]):
                        coords[i] = str(int(loc[i]))
                    elif isinstance(loc[i], np.ndarray) and loc[i].size == 1:
                        coords[i] = str(int(loc[i].flat[0]))
                    elif hasattr(loc[i], 'tolist'):
                        value = loc[i].tolist()
                        if isinstance(value, list) and len(value) == 1:
                            coords[i] = str(int(value[0]))
                except:
                    # Keep default value "0" if extraction fails
                    pass
            imdb_face_locs.append(",".join(coords))
        else:
            imdb_face_locs.append("0,0,0,0")
    except Exception as e:
        print(f"Error processing face location: {e}, type: {type(loc)}")
        imdb_face_locs.append("0,0,0,0")

for loc in wiki_face_location:
    try:
        # Simplified approach - just extract integers or use zeros
        if isinstance(loc, np.ndarray) and loc.size >= 4:
            coords = ["0", "0", "0", "0"]
            for i in range(4):
                try:
                    # Try different approaches to extract the value
                    if np.isscalar(loc[i]):
                        coords[i] = str(int(loc[i]))
                    elif isinstance(loc[i], np.ndarray) and loc[i].size == 1:
                        coords[i] = str(int(loc[i].flat[0]))
                    elif hasattr(loc[i], 'tolist'):
                        value = loc[i].tolist()
                        if isinstance(value, list) and len(value) == 1:
                            coords[i] = str(int(value[0]))
                except:
                    # Keep default value "0" if extraction fails
                    pass
            wiki_face_locs.append(",".join(coords))
        else:
            wiki_face_locs.append("0,0,0,0")
    except Exception as e:
        print(f"Error processing wiki face location: {e}")
        wiki_face_locs.append("0,0,0,0")

print(f"Processed {len(imdb_face_locs)} IMDB face locations")
print(f"Processed {len(wiki_face_locs)} Wiki face locations")

# Convert MATLAB datenum to Python date
def matlab_datenum_to_datetime(datenum):
    # MATLAB datenum starts from 0 at 0000-01-00, while Python's starts from 1 at 0001-01-01
    # MATLAB datenum 1 corresponds to 0001-01-01 in Python datetime
    # Need to adjust the difference in days accordingly
    days = datenum - 366  # 366 days difference between MATLAB's 0000-01-00 and Python's 0001-01-01
    return date.datetime.fromordinal(int(days)) + date.timedelta(days=days % 1)

# Format dates in human-readable format
imdb_formatted_dob = []
wiki_formatted_dob = []

for datenum in imdb_dob:
    try:
        dt = matlab_datenum_to_datetime(datenum)
        imdb_formatted_dob.append(dt.strftime('%Y-%m-%d'))
    except:
        imdb_formatted_dob.append('unknown')

for datenum in wiki_dob:
    try:
        dt = matlab_datenum_to_datetime(datenum)
        wiki_formatted_dob.append(dt.strftime('%Y-%m-%d'))
    except:
        wiki_formatted_dob.append('unknown')

# Calculate ages
imdb_age = []
wiki_age = []

for i in range(len(imdb_formatted_dob)):
    try:
        if imdb_formatted_dob[i] != 'unknown':
            d1 = date.datetime.strptime(imdb_formatted_dob[i], '%Y-%m-%d')
            d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years
        else:
            diff = -1
    except Exception as ex:
        print(f"Error calculating age: {ex}")
        diff = -1
    imdb_age.append(diff)

for i in range(len(wiki_formatted_dob)):
    try:
        if wiki_formatted_dob[i] != 'unknown':
            d1 = date.datetime.strptime(wiki_formatted_dob[i], '%Y-%m-%d')
            d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years
        else:
            diff = -1
    except Exception as ex:
        print(f"Error calculating age: {ex}")
        diff = -1
    wiki_age.append(diff)

# Create placeholder data for missing Wiki fields
wiki_names = ['unknown'] * len(wiki_age)
wiki_celeb_indices = [-1] * len(wiki_age)

# Stack the data for IMDB
print("Creating IMDB dataframe...")
final_imdb = np.vstack((
    imdb_age, 
    imdb_genders, 
    imdb_path, 
    imdb_names,
    imdb_formatted_dob,
    imdb_photo_taken,
    imdb_face_locs,
    imdb_face_score1, 
    imdb_face_score2,
    imdb_celeb_indices
)).T

# Stack the data for Wiki
print("Creating Wiki dataframe...")
final_wiki = np.vstack((
    wiki_age, 
    wiki_genders, 
    wiki_path,
    wiki_names,
    wiki_formatted_dob,
    wiki_photo_taken,
    wiki_face_locs,
    wiki_face_score1, 
    wiki_face_score2,
    wiki_celeb_indices
)).T

# Create DataFrames
final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

# Set column names
final_imdb_df.columns = cols
final_wiki_df.columns = cols

# Combine the DataFrames
print("Combining IMDB and Wiki dataframes...")
meta = pd.concat((final_imdb_df, final_wiki_df))

# Filter the data
meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']

# Create separate CSV files
print("Saving CSV files...")
# Full dataset with all fields
meta.to_csv('meta_full.csv', index=False)

# Create an IMDB-only CSV with all fields
final_imdb_df.to_csv('imdb_meta_full.csv', index=False)

# Create original format CSV for backward compatibility
meta_simplified = meta.drop(['face_score1', 'face_score2', 'name', 'dob', 'photo_taken', 'face_location', 'celeb_id'], axis=1)
meta_simplified = meta_simplified.sample(frac=1)  # Shuffle the data
meta_simplified.to_csv('meta.csv', index=False)

print("Done! CSV files created:")
print("- meta_full.csv (combined IMDB+Wiki with all fields)")
print("- imdb_meta_full.csv (IMDB-only with all fields)")
print("- meta.csv (original format for compatibility)")
