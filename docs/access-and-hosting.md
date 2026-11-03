# Access and hosting

Orbital is a Python/Streamlit application. A browser connects to the Python
process; static website hosting alone cannot run simulations or data refreshes.
No public website, cloud deployment, domain or tunnel is configured by this
release.

## Local Mac

From the repository root, activate the environment used to install
`requirements.txt`, then start a loopback-only preview:

```sh
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` on the same computer. That address does not identify
the Mac when opened on a phone. Keep the Mac awake while using the app.

## Phone on the same Wi-Fi

Find the Mac's Wi-Fi address in System Settings → Wi-Fi → Details → TCP/IP.
On Macs where Wi-Fi is `en0`, `ipconfig getifaddr en0` also prints it.
Start a separate instance using that address:

```sh
python -m streamlit run streamlit_app.py --server.address <MAC_WIFI_IP> --server.port 8509
```

Open `http://<MAC_WIFI_IP>:8509` on the phone. Replace the angle-bracket
placeholder before running the command. Both devices must be on a network that
allows devices to communicate; guest-network isolation or a firewall can prevent
access. The address may change after reconnecting to Wi-Fi.

The app currently has no login. A LAN listener exposes its data refresh,
simulation and state-save controls to devices that can reach that
address. Enable this only on a trusted network with the owner's approval.
Do not forward the port through the router for public access. Normal dashboard
refreshes can start background manager jobs; state still advances only after the
existing explicit confirmation. Do not run competing updates from two sessions.

## A personal website

A possible future layout is a portfolio at the root domain and Orbital at
`fpl.<your-domain>`. The portfolio can link to the app. The app needs a Python
runtime, durable data/model/job storage, backups, a background-job recovery plan,
HTTPS and an access boundary around management actions.

Free options discussed, not deployed:

- [Cloudflare Pages](https://developers.cloudflare.com/pages/platform/limits/)
  can host the portfolio; its free plan supports custom domains. It does not
  provide the existing Streamlit server process.
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/routing-to-tunnel/)
  can publish the Mac-hosted app through a hostname. The Mac and tunnel must
  remain online. Add an identity policy, such as
  [Cloudflare Access](https://www.cloudflare.com/sase/products/access/), before
  exposing management controls. A public read-only showcase would need a
  separate application permission boundary.
- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
  offers free hosting. Its
  [resource limits and app sleeping](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
  need evaluation for heavy simulations and scheduled work; local season state
  must be moved to durable storage before relying on it.
- [Oracle Always Free](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
  includes eligible compute/storage, but capacity and idle-instance reclamation
  rules apply. Deployment compatibility and workload fit have not been tested.
- [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
  can host a static portfolio and custom domain; it cannot run this Python
  backend. Merely pushing the repository does not deploy Orbital.

Check current provider terms before choosing a service. Domain registration,
home electricity/internet, and optional Astra API calls are separate from free
hosting. Keep API credentials in the server environment or private secrets
configuration, never in Git or browser code. Before moving an in-progress
manager, back up and verify the existing state, then designate one place to run
live updates so the two copies do not diverge.
