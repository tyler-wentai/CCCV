# Note: We grab and save the cshapes data from R because it contains
# different column data than the files directly on the cshape website:
# https://icr.ethz.ch/data/cshapes/
# Most importantly, the cshape data.frame from R contains the country
# status, which we desire, since we are looking only at independent
# countries (same as Hsiang et al. 2011).

library(cshapes)
library(sf)

packageVersion('cshapes')

# load cshape data.frame
dat <- cshp(date = NA, useGW = TRUE, dependencies = FALSE)

# Write file
path = '/Users/tylerbagwell/Desktop/cshape_files/cshpR.shp'
st_write(dat, path)


