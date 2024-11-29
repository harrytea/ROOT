str1 = '''
Please determine the hierarchical relationships between the objects ('''


# str2 = ''') marked as point in the image. Use only these four hierarchical relationships: support, contain, attach, and hang.

# For example, use "support" for objects on a table or chair, "contain" for objects inside a bookshelf or bottle, and "hang" for objects on the wall like doors, curtains, or paintings. Objects on the ceiling, such as lights, should use "attach". If there's a drawer in a table or objects inside the drawer, the relationship should be "contain". For objects on the floor, like tables on a carpet, the relationship is 'floor supports rug supports table'.

# Present the relationships in a list format, with the ceiling, wall, floor must include. For the object that have no children, just include the object name. Here's an example\n ```list\n[['floor', 'support', 'object1'], ['object1', 'support', 'object2'], ['object2', 'support', 'object3'], 'object3', ['object1', 'support', 'object4'], 'object4', ['floor', 'support', 'object5'], 'object5', 'ceiling', 'wall']\n```\n
# '''


str2 = ''') marked as point in the image. Use only these four hierarchical relationships: support, contain, attach, and hang.

For example, use "support" for objects on a table or chair, "contain" for objects inside a bookshelf or bottle, and "hang" for objects on the wall like doors, curtains, or paintings. Objects on the ceiling, such as lights, should use "attach". If there's a drawer in a table or objects inside the drawer, the relationship should be "contain". For objects on the floor, like tables on a carpet, the relationship is 'floor supports rug supports table'.

Present the relationships in a JSON tree format, with the ceiling, wall, floor as the root nodes. Here's an example JSON structure:

```json
{
    "ceiling": {
        "attach": [
            {
                "object": {}
            }
        ]
    }, 
    "wall": {}, 
    "floor": {
        "support": [
            {
                "object": {
                    "support": [
                        {
                            "object": {
                                "support": [
                                    {
                                        "object": {}
                                    }
                                ]
                            }
                        }, 
                        {
                            "object": {}
                        }
                    ]
                }
            },
            {
                "object": {}
            }
        ]
    }
}
```
'''




str_human = "Present the relationships in a JSON tree format, with the ceiling, wall, floor as the root nodes. Here's an example JSON structure:\n\n```json\n{\n    \"ceiling\": {\n        \"attach\": [\n            {\n                \"object\": {}\n            }\n        ]\n    }, \n    \"wall\": {}, \n    \"floor\": {\n        \"support\": [\n            {\n                \"object\": {\n                    \"support\": [\n                        {\n                            \"object\": {\n                                \"support\": [\n                                    {\n                                        \"object\": {}\n                                    }\n                                ]\n                            }\n                        }, \n                        {\n                            \"object\": {}\n                        }\n                    ]\n                }\n            },\n            {\n                \"object\": {}\n            }\n        ]\n    }\n}\n```\n"


str_human_replace = "Present the relationships in a list format, with the ceiling, wall, floor must include. For the object that have no children, just include the object name. Here's an example\n ```list\n[['floor', 'support', 'object1'], ['object1', 'support', 'object2'], ['object2', 'support', 'object3'], 'object3', ['object1', 'support', 'object4'], 'object4', ['floor', 'support', 'object5'], 'object5', 'ceiling', 'wall']\n```\n"