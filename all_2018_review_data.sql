CREATE TABLE IF NOT EXISTS businesses_2018_all_data (business_id TEXT PRIMARY KEY NOT NULL,
                                                      name TEXT,
                                                      address TEXT,
                                                      city TEXT,
                                                      state TEXT, 
                                                      postal_code TEXT,
                                                      latitude REAL,
                                                      longitude REAL,
                                                      stars REAL,
                                                      review_count INTEGER,
                                                      is_open INTEGER,
                                                      attributes TEXT,
                                                      categories TEXT,
                                                      hours TEXT);


UPDATE businesses_2018_all_data
SET 
    business_id = business_id,
    name = (SELECT all_businesses.name FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    address = (SELECT all_businesses.address FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    city = (SELECT all_businesses.city FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    state = (SELECT all_businesses.state FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    postal_code = (SELECT all_businesses.postal_code FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    latitude = (SELECT all_businesses.latitude FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    longitude = (SELECT all_businesses.longitude FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    stars = (SELECT all_businesses.stars FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    review_count = (SELECT all_businesses.review_count FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    is_open = (SELECT all_businesses.is_open FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    attributes = (SELECT all_businesses.attributes FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    categories = (SELECT all_businesses.categories FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id),
    hours = (SELECT all_businesses.hours FROM all_businesses WHERE all_businesses.business_id = businesses_2018_all_data.business_id);
    






-- SELECT * FROM all_businesses;
-- SELECT * FROM businesses_2018;
-- SELECT * FROM businesses_2018_all_data;

SELECT city, COUNT(city) AS count FROM businesses_2018_all_data GROUP BY city ORDER BY count DESC;


