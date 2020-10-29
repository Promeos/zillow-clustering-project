# Reconstruction in Progress
I need to perform a complete revamp of this repositry: README, notebooks, function files, and summary notebook. I failed to address the __main__ purpose of this project. In its current state you'll have to dig to find insights, which is not user-friendly nor professional. If you are interested in this project please star it and comeback in a few days.

## Zillow Clustering Project
Using single-unit/single-family residential properties sold in the Southern California area,  predict Zillow's 2017 log error drivers. Use KMeans clustering techniques to uncover drivers of log error.

# Summary
## _Work in progress_

## Data Dictionary
| Features | Description |
| :------ | :---------- |
| `latitude` | Latitude of the middle of the parcel divided by 10e6 |
| `longitude` | Longitude of the middle of the parcel divided by 10e6 |
| `fips` | Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details |
| `parcelid` | Unique identifier for parcels (lots) |
| `basement_area_sqft` | Finished living area below or partially below ground level |
| `num_of_bedrooms` | Number of bedrooms in home |
| `num_of_restrooms` | Number of bathrooms in home including fractional bathrooms  |
| `livable_area_sqft` | Living room square feet | 
| `lot_size_sqft` | Area of the lot in square feet |
| `year_built` | The Year the principal residence was built |
| `has_basement` | If the home has a basement |
| `basement_area_sqft` | The area of the basement |
| `has_hottub_orspa` | Does the home have a hot tub or spa |
| `has_pool` | If the home has a pool |
| `pool_area_sqft` | Area of pool in square feet |
| `has_patio` | If the home has a patio in yard |
| `patio_sqft` | Pool without hot tub |
| `has_shed` | If the home has a storage shed/building in yard |
| `property_tax` |The total property tax assessed for that assessment year |
| `taxable_value` | The total tax assessed value of the parcel |
| `structure_tax` | The assessed value of the built structure on the parcel |
| `land_tax` |The assessed value of the land area of the parcel |

| Target | Description |
| :------ | :---------- |
| `logerror` | The log difference between Zillow's Zestimate and the property sale price |


## Project Organization
### _Work in progress_
```
├── README.md           <- The top-level README for developers using this project.
│
│
├── Report.ipynb           <- The main notebook for the project
```

## Requirements
- numpy >= 1.1.2
- pandas >= 1.18.1
- scipy >=1.4.1
- sklearn >= 0.23.2
- matplotlib >= 3.3.1
- seaborn >= 0.11.0

## Setup
1. Download a zip file of the repository [here](https://github.com/Promeos/zillow-clustering-project/archive/main.zip)
2. Clone this repository using:
```
git clone git@github.com:Promeos/zillow-clustering-project.git
```

To open the file in a jupyter notebook use following code:
``` python
import pandas as pd
df = pd.read_csv('.data/zillow.csv')
```

## Planning
### __Work in progress__
Goals:
1. The goal of this project is to discover the drivers of Zillow's Zestimate logerror using single unit properties sold in May and June of 2017.

2. Generate valuable insights in EDA. Test, Visualize, Document.

3. Build a model to accurately predict ...

Measures of Success:
<strong>DOCUMENT EVERYTHING</strong>
1. Clearly explain the acquisition and preparation stages. Why certain observations/values/columns are removed, imputing values, casting etc. Document outliers and how they are handled.
2. Discover 2 features that drive logerror using statistical tests.
    - 1 Chi2 Test
    - 2/3 T-tests
    - 1 ANOVA
3. Use kmeans clustering to create new features from the Zillow dataset.
    - Determine if the new features created with kmeans clustering reduces the logerror of the model.
    - MVP: 3 Clustering Features.
    - Document what the new features extrapolated from the data.
4. Create 4 distinct models
    - Outperform the baseline model.
    - Write a short summary of how the model works.
        - Features used
        - Preprocessing methods
        - Algorithm used
        - Hyperparameters used
        - Train, validate, and test splits
        - Score/Performance

Your audience for this project is a data science team. The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

MVP/Final Product:
1. A jupyter notebook that incorporates each stage of the pipeline.
    - Clean documentation, docstrings
    - Separate .py files to store functions used in acquisition/preparation/modeling
    - Clean code
    - Separate .ipynb's for acqusition and preparation process.
        - Include summary of acquistion and preparation in final notebook.
        - Include links to each notebook.
2. Deliver to a data science team.
3. The insights from the notebook will be used to uncover drivers of Zillow's Zestimate error.
4. The project is finished if I can explain the drivers of Zillow's Zestimate error using statistical tests and outperform the baseline logerror.
5. Rehearse presenting the notebook at least 5 times.
    - Ask questions/answer with notebook
    - clearly state what each graph means
    - clearly state what the takeaways are in each section
    - simplify graphs and reiterate key points
    - present what is useful

Hypothesis Tests:

## Acknowledgements
- Codeup Data Science Team
- Darden Cohort

## Contact
- [@Promeos42](https://twitter.com/Promeos42)
- 📫 christopher.logan.ortiz@gmail.com

