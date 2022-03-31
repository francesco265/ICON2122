# -*- coding: utf-8 -*-
import ssl
from SPARQLWrapper import SPARQLWrapper, JSON

ssl._create_default_https_context = ssl._create_unverified_context
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
QUERY_DENSITY = "?city dbo:populationDensity ?density."
QUERY_AREAPOP = """ ?city dbo:populationTotal ?population.
                    ?city dbo:areaLand ?area."""
QUERY_COORDS  = """ ?city geo:lat ?lat.
                    ?city geo:long ?long."""

def _query(city: str, snippet: str):
    sparql.setQuery("""
        SELECT * WHERE {
            {
                SELECT ?city WHERE {
                    ?city dbo:subdivision dbr:Washington_\(state\).
                    FILTER(CONTAINS(str(?city), "%s")).
                }
                ORDER BY STRLEN(str(?city))
                LIMIT 1    
            }
            %s
        }
        LIMIT 1
        """ % (city.replace(" ", "_"), snippet))
    if len(sparql.queryAndConvert()["results"]["bindings"]) > 0:
        return sparql.queryAndConvert()["results"]["bindings"][0]
    else:
        return None

def citiesDensity(cities: list) -> dict:
    """
    Restitusice un dizionario contenente la densità di popolazione per le città
    trovate su DBPedia
    """
    densities = dict()
    for city in cities:
        x = _query(city, QUERY_DENSITY)
        if x != None:
            densities[city] = float(x["density"]["value"])
        else:
            y = _query(city, QUERY_AREAPOP)
            if y != None:
                densities[city] = float(y["population"]["value"]) / (float(y["area"]["value"]) / 1e6)
    return densities

def citiesCoords(cities: list) -> dict:
    coords = dict()
    for city in cities:
        coord = _query(city, QUERY_COORDS)
        if coord != None:
            coords[city] = [float(coord["lat"]["value"]), float(coord["long"]["value"])]
    return coords