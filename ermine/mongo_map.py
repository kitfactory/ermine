from pymongo import MongoClient

client = MongoClient("localhost")
db = client.test

def get_map_keys(map_collection:str):
    obj = db[map_collection].find_one({})
    keys = []
    for k in obj:
        if (k != "_id")and(k!="label") :
            keys.append(k)
    print(keys)
    return keys

def map_metadata(metadata_collection, map_collection,keys):
    lets = {}
    and_expr = []
    for k in keys:
        lets["label_"+k] = "$"+k
        and_expr.append({"$eq":["$$label_"+k,"$"+k]})

    pipeline = [
        {"$lookup":{
            "from":map_collection,
            "let":lets,
            "pipeline":[
                {"$match":
                    {"$expr":
                        {"$and":and_expr}
                    }                
                }
            ],
            "as":"result"
        }},
        {"$unwind":"$result"},
        {"$addFields":{"label":"$result.label"}},
        {"$project":{"result":0}}
    ]
    mapped = db[metadata_collection].aggregate(pipeline)
    for m in mapped:
        print(m)

keys = get_map_keys("b")
map_metadata("a","b",keys)
