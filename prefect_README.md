
## Install
```bash
pip install -U prefect
```

## Allow port 4200 for Zerotier connection
win + r:
```bash
 #opens Windows Defendeer Firewall Advanced
wf.msc 
```

```
1.
In the left pane choose Inbound Rules, then in the right pane click New Rule…

2.
Select Port → Next

3.
Choose TCP, enter 4200 in “Specific local ports” → Next

4.
Select Allow the connection → Next

5.
Check Private (and Domain if you use a corporate domain) → Next

6.
Give it a name like “Prefect Orion UI” → Finish

```

## Run on localhost / Start prefect
First save the prefect api url to work with the zerotier ip

```bash
prefect config set PREFECT_API_URL="http://<Your-PC’s-ZeroTier-IP>:4200/api"
```
```bash
prefect server start --host 0.0.0.0 --port 4200
```

## Connect to Prefect
Connect machine to zerotier then:

``http://<Your-PC’s-ZeroTier-IP>:4200``

or locally on the pc with:
```bash
localhost:4200
-or
<PC´s ip>:4200
```