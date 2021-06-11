SELECT
    date,
    avg(precipitation),
    avg(temperature),
    avg(evaporation),
    avg(surface_runoff),
    coordinates.river_id
FROM
    variables
INNER JOIN
    coordinates
ON
    variables.coordinate_id = coordinates.id
GROUP BY
    date,
    coordinates.river_id;