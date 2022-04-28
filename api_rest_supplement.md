```python
d = {}

d['signature_name'] = 'serving_default'
d['instances'] = [{'examples': {'b64': base64.b64encode(example_proto).decode('utf-8')}}, 
                  {'examples': {'b64': base64.b64encode(example_proto).decode('utf-8')}}]
```


# column format
```python
d['signature_name'] = 'serving_default'
d['inputs'] = {'examples': [{'b64': base64.b64encode(example_proto).decode('utf-8')}, 
                            {'b64': base64.b64encode(example_proto).decode('utf-8')}]
              }
