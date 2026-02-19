import streamlit as st
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import json
import os

try:
    import folium
    from streamlit_folium import st_folium
    HAS_MAP = True
except ImportError:
    HAS_MAP = False

# ============================================================
# PERZISTENTNÍ PAMĚŤ
# ============================================================
HISTORY_FILE = "geodetic_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Historii se nepodařilo uložit: {e}")

# ============================================================
# INICIALIZACE SESSION STATE
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'history' not in st.session_state:
    st.session_state.history = load_history()

# ============================================================
# NAVIGACE
# ============================================================
def go_to_home():      st.session_state.page = 'home'
def go_to_hgu1():      st.session_state.page = 'hgu1'
def go_to_hgu2():      st.session_state.page = 'hgu2'
def go_to_dilcove():   st.session_state.page = 'dilcove'
def go_to_history():   st.session_state.page = 'history'
def go_to_prevodnik(): st.session_state.page = 'prevodnik'

def clear_inputs():
    keys_zeros = ['ea1','na1','alta1','s1','ang1','pol1',
                  'ea2','na2','alta2','eb2','nb2','altb2']
    keys_none  = ['dil_m','dil_km','dil_dc']
    for k in keys_zeros:
        if k in st.session_state: st.session_state[k] = 0
    for k in keys_none:
        if k in st.session_state: st.session_state[k] = None

def clear_history():
    st.session_state.history = []
    save_history([])

# ============================================================
# VALIDACE SMĚRNÍKU
# ============================================================
def validate_smernik(value, label="Směrník"):
    if not (0 <= value <= 5999):
        st.error(
            f"⚠️ Chyba: **{label}** musí být v rozsahu **0–5999 dc**! "
            f"(Zadáno: {value})"
        )
        st.stop()

# ============================================================
# ČISTÁ MATEMATIKA: UTM → WGS84 (bez pyproj)
# ============================================================
def utm_to_wgs84_math(easting, northing, zone_number, northern=True):
    """Převod UTM souřadnic na WGS84 – čistá matematika, žádné závislosti."""
    a   = 6378137.0
    f   = 1 / 298.257223563
    e2  = 2 * f - f ** 2
    ep2 = e2 / (1 - e2)
    k0  = 0.9996

    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0

    lon_origin = (zone_number - 1) * 6 - 180 + 3

    M    = y / k0
    mu   = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1   = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = (mu
            + (3*e1/2 - 27*e1**3/32)      * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96)               * math.sin(6*mu)
            + (1097*e1**4/512)             * math.sin(8*mu))

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = ep2 * math.cos(phi1)**2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1)**2)**1.5
    D  = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
          D**2/2
        - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*ep2)                    * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*ep2 - 3*C1**2)    * D**6/720)

    lon = (D
           - (1 + 2*T1 + C1)                                        * D**3/6
           + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*ep2 + 24*T1**2)      * D**5/120
          ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon) + lon_origin

def wgs84_to_utm_math(lat_deg, lon_deg):
    """Převod WGS84 na UTM – čistá matematika."""
    a   = 6378137.0
    f   = 1 / 298.257223563
    e2  = 2 * f - f ** 2
    ep2 = e2 / (1 - e2)
    k0  = 0.9996

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    zone_number  = int((lon_deg + 180) / 6) + 1
    lon_origin   = math.radians((zone_number - 1) * 6 - 180 + 3)
    zone_letter  = "CDEFGHJKLMNPQRSTUVWXX"[int((lat_deg + 80) / 8)]

    N  = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    T  = math.tan(lat)**2
    C  = ep2 * math.cos(lat)**2
    A  = math.cos(lat) * (lon - lon_origin)
    M  = a * (
          (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256)   * lat
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024)   * math.sin(2*lat)
        + (15*e2**2/256 + 45*e2**3/1024)           * math.sin(4*lat)
        - (35*e2**3/3072)                           * math.sin(6*lat))

    easting  = (k0 * N * (A + (1-T+C)*A**3/6
                + (5-18*T+T**2+72*C-58*ep2)*A**5/120) + 500000.0)
    northing = (k0 * (M + N*math.tan(lat) * (
                A**2/2 + (5-T+9*C+4*C**2)*A**4/24
                + (61-58*T+T**2+600*C-330*ep2)*A**6/720)))
    if lat_deg < 0:
        northing += 10000000.0

    return easting, northing, zone_number, zone_letter

# ============================================================
# PŘEVOD MGRS → WGS84 (čistá matematika)
# ============================================================
def validate_zone_square(zone, square):
    zone   = zone.strip().upper()
    square = square.strip().upper()
    if len(zone) != 3 or not zone[:2].isdigit() or not zone[2].isalpha():
        return None, "Zóna musí mít formát: 2 číslice + 1 písmeno (např. **33U**)."
    if len(square) != 2 or not square.isalpha():
        return None, "100km čtverec musí mít formát: 2 písmena (např. **VR**)."
    return zone + square, None

def mgrs_en_to_wgs84(e, n, zone_square):
    try:
        zone_num    = int(zone_square[:2])
        zone_letter = zone_square[2].upper()
        sq_e        = zone_square[3].upper()
        sq_n        = zone_square[4].upper()

        set_num = (zone_num - 1) % 3
        if set_num == 0:
            e_letters = "ABCDEFGH"
        elif set_num == 1:
            e_letters = "JKLMNPQR"
        else:
            e_letters = "STUVWXYZ"

        e_idx        = e_letters.index(sq_e)
        utm_easting  = (e_idx + 1) * 100000 + (int(e) % 100000)

        n_letters         = "ABCDEFGHJKLMNPQRSTUV"
        n_offset          = 5 if zone_num % 2 == 0 else 0
        n_letters_shifted = (n_letters * 3)[n_offset:]
        n_idx             = n_letters_shifted.index(sq_n)
        utm_northing      = n_idx * 100000 + (int(n) % 100000)

        band_northings = {
            'C': 1000000,  'D': 2000000,  'E': 3000000,  'F': 4000000,
            'G': 5000000,  'H': 6000000,  'J': 7000000,  'K': 8000000,
            'L': 9000000,  'M': 10000000, 'N': 0,        'P': 1000000,
            'Q': 2000000,  'R': 3000000,  'S': 4000000,  'T': 5000000,
            'U': 6000000,  'V': 7000000,  'W': 8000000,  'X': 9000000,
        }
        min_northing = band_northings.get(zone_letter, 0)
        while utm_northing < min_northing:
            utm_northing += 2000000

        northern = zone_letter >= 'N'
        lat, lon = utm_to_wgs84_math(utm_easting, utm_northing,
                                     zone_num, northern)
        return lat, lon
    except Exception:
        return None, None

# ============================================================
# NÁČRT SITUACE (matplotlib)
# ============================================================
def draw_plot(ea, na, eb, nb, angle_dilce, distance_m):
    fig, ax = plt.subplots(figsize=(6, 6))
    de = eb - ea
    dn = nb - na

    margin    = max(distance_m * 0.3, 300)
    x_min, x_max = min(ea, eb) - margin, max(ea, eb) + margin
    y_min, y_max = min(na, nb) - margin, max(na, nb) + margin
    max_range = max(x_max - x_min, y_max - y_min)
    x_center  = (x_min + x_max) / 2
    y_center  = (y_min + y_max) / 2

    xlim = (x_center - max_range / 2, x_center + max_range / 2)
    ylim = (y_center - max_range / 2, y_center + max_range / 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.plot([ea, eb], [na, nb],
            color='red', marker='o', linestyle='-',
            linewidth=2, markersize=8, zorder=3)
    ax.annotate('A', (ea, na), textcoords="offset points",
                xytext=(-15, -15), ha='center', fontsize=12, fontweight='bold')
    ax.annotate('B', (eb, nb), textcoords="offset points",
                xytext=(15, 15), ha='center', fontsize=12, fontweight='bold')

    north_len = max_range * 0.16
    ax.plot([ea, ea], [na, na + north_len],
            color='gray', linestyle='--', linewidth=1.3, zorder=1, alpha=0.85)
    ax.annotate('',
                xy=(ea, na + north_len),
                xytext=(ea, na + north_len * 0.82),
                arrowprops=dict(arrowstyle='->', color='gray',
                                lw=1.5, mutation_scale=14))

    mid_e  = (ea + eb) / 2
    mid_n  = (na + nb) / 2
    offset = max_range * 0.05
    ax.text(mid_e + offset, mid_n,
            f"ΔE = {de:+.0f} m\nΔN = {dn:+.0f} m",
            ha='left', va='center', fontsize=9, color='navy',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
                      alpha=0.92, edgecolor='gray', linewidth=0.8))

    angle_int = int(round(angle_dilce)) % 6000
    angle_str = f"σ = {angle_int // 100:02d}-{angle_int % 100:02d} dc"
    dist_str  = f"d = {distance_m / 1000.0:.3f} km"

    ax.text(0.97, 0.97, angle_str, transform=ax.transAxes,
            color='crimson', fontsize=11, fontweight='bold', ha='right', va='top')
    ax.text(0.97, 0.90, dist_str, transform=ax.transAxes,
            color='steelblue', fontsize=11, fontweight='bold', ha='right', va='top')

    ax.annotate('',
                xy=(0.065, 0.963), xytext=(0.065, 0.915),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=2.2, mutation_scale=16))
    ax.text(0.065, 0.975, 'S', transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', va='bottom', color='black')

    formatter = ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("E [km]", fontweight='bold')
    ax.set_ylabel("N [km]", fontweight='bold', rotation=0, labelpad=20)
    ax.set_title("Náčrt situace", fontweight='bold', fontsize=13)

    plt.tight_layout()
    return fig

# ============================================================
# INTERAKTIVNÍ MAPA (folium)
# ============================================================
def show_map(lat_a, lon_a, lat_b, lon_b, label_a, label_b, map_key="map"):
    if not HAS_MAP:
        st.error("Nainstalujte: `pip install folium streamlit-folium`")
        return

    center_lat = (lat_a + lat_b) / 2
    center_lon = (lon_a + lon_b) / 2

    dist_deg = math.sqrt((lat_b - lat_a)**2 + (lon_b - lon_a)**2)
    if dist_deg < 0.005:   zoom = 16
    elif dist_deg < 0.02:  zoom = 14
    elif dist_deg < 0.1:   zoom = 12
    elif dist_deg < 0.5:   zoom = 10
    else:                  zoom = 8

    tile_layer = st.selectbox(
        "Typ mapové vrstvy:",
        ["OpenStreetMap", "OpenTopoMap", "Esri Satellite"],
        key=f"tile_{map_key}"
    )

    tile_urls = {
        "OpenStreetMap": {
            "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attr":  "© OpenStreetMap contributors",
        },
        "OpenTopoMap": {
            "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr":  "© OpenTopoMap contributors",
        },
        "Esri Satellite": {
            "tiles": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            "attr":  "© Esri",
        },
    }

    t = tile_urls[tile_layer]
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=t["tiles"],
        attr=t["attr"],
    )

    folium.Marker(
        location=[lat_a, lon_a],
        tooltip="Bod A",
        popup=folium.Popup(f"<b>Bod A</b><br>{label_a}", max_width=220),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    folium.Marker(
        location=[lat_b, lon_b],
        tooltip="Bod B",
        popup=folium.Popup(f"<b>Bod B</b><br>{label_b}", max_width=220),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.PolyLine(
        locations=[[lat_a, lon_a], [lat_b, lon_b]],
        color="red", weight=2.5, dash_array="6",
        tooltip="Spojnice A–B",
    ).add_to(m)

    st_folium(m, use_container_width=True, height=420,
              returned_objects=[], key=f"map_{map_key}")

# ============================================================
# WIDGET: MGRS zóna a 100km čtverec
# ============================================================
def mgrs_zone_input(key_suffix):
    st.markdown("**Zadejte MGRS identifikátor oblasti** *(platí pro oba body)*")
    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    with c1:
        zone   = st.text_input("Zóna", value="33U",
                               placeholder="např. 33U",
                               key=f"mgrs_zone_{key_suffix}")
    with c2:
        square = st.text_input("100km čtverec", value="",
                               placeholder="např. VR",
                               key=f"mgrs_sq_{key_suffix}")
    with c3:
        st.markdown(
            "<br><small>Příklad: <b>33U VR 12400 32700</b><br>"
            "→ Zóna: <b>33U</b> | Čtverec: <b>VR</b></small>",
            unsafe_allow_html=True
        )
    if not zone and not square:
        return None, None
    zone_square, err = validate_zone_square(zone, square)
    return zone_square, err

# ============================================================
# POMOCNÁ FUNKCE DMS
# ============================================================
def to_dms(deg, is_lat):
    direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    val = abs(deg)
    d   = int(val)
    md  = (val - d) * 60
    mi  = int(md)
    sd  = (md - mi) * 60
    return f"{d}° {mi}' {sd:.2f}\" {direction}"

# ============================================================
# STRÁNKA: HLAVNÍ MENU
# ============================================================
if st.session_state.page == 'home':
    st.title("Dělostřelecká kalkulačka")
    st.markdown("---")
    st.write("**Vyberte úlohu, kterou chcete počítat:**")
    st.button("HGÚ 1",              on_click=go_to_hgu1,      use_container_width=True)
    st.button("HGÚ 2",              on_click=go_to_hgu2,      use_container_width=True)
    st.button("Dílcové pravidlo",   on_click=go_to_dilcove,   use_container_width=True)
    st.button("Převodník jednotek", on_click=go_to_prevodnik, use_container_width=True)
    st.button("Historie výpočtů",   on_click=go_to_history,   use_container_width=True)

# ============================================================
# STRÁNKA: HISTORIE VÝPOČTŮ
# ============================================================
elif st.session_state.page == 'history':
    st.title("Historie výpočtů")
    st.button("Zpět na hlavní menu", on_click=go_to_home, use_container_width=True)
    st.markdown("---")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.button("Vymazat historii", on_click=clear_history, use_container_width=True)
    else:
        st.info("Zatím nebyly provedeny žádné výpočty.")

# ============================================================
# STRÁNKA: PŘEVODNÍK JEDNOTEK
# ============================================================
elif st.session_state.page == 'prevodnik':
    st.title("Převodník jednotek")
    st.button("Zpět na hlavní menu", on_click=go_to_home, use_container_width=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["Úhly", "Souřadnice"])

    with tab1:
        st.subheader("Převod úhlových měr")
        uhl_vstup    = st.number_input("Zadejte hodnotu úhlu:", value=0.0, step=1.0)
        uhl_jednotka = st.selectbox("Z jaké jednotky převádíte?",
                                    ["Dílce (dc - 6000)", "NATO Mils (mil - 6400)", "Stupně (°)"])

        if st.button("Převést úhly", type="primary", use_container_width=True):
            if uhl_jednotka == "Dílce (dc - 6000)":
                dc, mils, deg = uhl_vstup, uhl_vstup*(6400/6000), uhl_vstup*(360/6000)
            elif uhl_jednotka == "NATO Mils (mil - 6400)":
                mils, dc, deg = uhl_vstup, uhl_vstup*(6000/6400), uhl_vstup*(360/6400)
            else:
                deg, dc, mils = uhl_vstup, uhl_vstup*(6000/360), uhl_vstup*(6400/360)

            st.success("Převod úhlů byl úspěšný!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Dílce (6000)",     f"{dc:.2f} dc")
            c2.metric("NATO Mils (6400)", f"{mils:.2f} mil")
            c3.metric("Stupně (360)",     f"{deg:.2f}°")
            zapis = (f"{uhl_vstup} {uhl_jednotka.split(' ')[0]} = "
                     f"{dc:.2f} dc | {mils:.2f} mil | {deg:.2f}°")
            st.session_state.history.append({"Úloha": "Převod úhlů", "Zápis": zapis})
            save_history(st.session_state.history)

    with tab2:
        st.subheader("Převod souřadnic")
        typ_vstupu = st.radio("Směr převodu:",
                              ["UTM → WGS84", "WGS84 → UTM"])

        if typ_vstupu == "UTM → WGS84":
            c1, c2 = st.columns(2)
            with c1:
                utm_zone = st.number_input("Zóna:", min_value=1, max_value=60,
                                           value=33, step=1)
                utm_hemi = st.selectbox("Polokoule:", ["Severní (N)", "Jižní (S)"])
            with c2:
                utm_e = st.number_input("East (E):", value=0.0, step=1.0)
                utm_n = st.number_input("North (N):", value=0.0, step=1.0)

            if st.button("Převést", type="primary", use_container_width=True):
                try:
                    is_n     = "Severní" in utm_hemi
                    lat, lon = utm_to_wgs84_math(utm_e, utm_n, utm_zone, is_n)
                    st.success("Převod byl úspěšný!")
                    st.write(f"**UTM:** Zóna {utm_zone}, E: {utm_e:.0f}, N: {utm_n:.0f}")
                    st.write(f"**WGS84:** Lat: {lat:.6f}°, Lon: {lon:.6f}°")
                    st.write(f"**DMS:** {to_dms(lat, True)}, {to_dms(lon, False)}")
                    zapis = (f"UTM {utm_zone} E:{utm_e:.0f} N:{utm_n:.0f} ➔ "
                             f"Lat:{lat:.5f} Lon:{lon:.5f}")
                    st.session_state.history.append({"Úloha": "Převod UTM", "Zápis": zapis})
                    save_history(st.session_state.history)
                except Exception as ex:
                    st.error(f"Chyba: {ex}")

        else:
            c1, c2 = st.columns(2)
            with c1:
                lat_in = st.number_input("Zeměpisná šířka:", value=0.0, format="%.6f")
            with c2:
                lon_in = st.number_input("Zeměpisná délka:", value=0.0, format="%.6f")

            if st.button("Převést", type="primary", use_container_width=True):
                try:
                    e, n, zn, zl = wgs84_to_utm_math(lat_in, lon_in)
                    st.success("Převod byl úspěšný!")
                    st.write(f"**WGS84:** Lat: {lat_in:.6f}°, Lon: {lon_in:.6f}°")
                    st.write(f"**UTM:** Zóna {zn}{zl}, E: {e:.0f}, N: {n:.0f}")
                    st.write(f"**DMS:** {to_dms(lat_in, True)}, {to_dms(lon_in, False)}")
                    zapis = (f"Lat:{lat_in:.4f} Lon:{lon_in:.4f} ➔ "
                             f"UTM {zn}{zl} E:{e:.0f} N:{n:.0f}")
                    st.session_state.history.append({"Úloha": "Převod WGS84", "Zápis": zapis})
                    save_history(st.session_state.history)
                except Exception as ex:
                    st.error(f"Chyba: {ex}")

# ============================================================
# STRÁNKA: DÍLCOVÉ PRAVIDLO
# ============================================================
elif st.session_state.page == 'dilcove':
    st.title("Dílcové pravidlo")
    st.button("Zpět na hlavní menu",     on_click=go_to_home,   use_container_width=True)
    st.button("Vymazat všechny hodnoty", on_click=clear_inputs, use_container_width=True)
    st.markdown("---")
    st.subheader("Zadání hodnot")
    st.write("Zadejte přesně **dvě známé hodnoty**. Třetí políčko nechte prázdné.")

    col1, col2, col3 = st.columns(3)
    with col1: dil_m  = st.number_input("Velikost / Výška (m):", min_value=0.0,
                                         step=1.0, value=None, key='dil_m')
    with col2: dil_km = st.number_input("Vzdálenost (km):",      min_value=0.0,
                                         step=0.1, value=None, key='dil_km')
    with col3: dil_dc = st.number_input("Úhel (dc):",            min_value=0.0,
                                         step=0.1, value=None, key='dil_dc')

    if st.button("Vypočítat", type="primary", use_container_width=True):
        vyplnene = [v for v in [dil_m, dil_km, dil_dc] if v is not None]
        if len(vyplnene) != 2:
            st.error("Chyba: Vyplňte přesně 2 políčka!")
        else:
            zapis = None
            if dil_m is None:
                res = dil_km * dil_dc * 1.05
                st.success("**Výpočet byl úspěšný!**")
                st.metric("Velikost / Výška (m)", f"{res:.1f} m")
                zapis = f"m = {dil_km:g} km × {dil_dc:g} dc (+5%) = {res:.1f} m"
            elif dil_km is None:
                if dil_dc == 0:
                    st.error("Úhel nesmí být nulový!")
                else:
                    res = (dil_m / dil_dc) * 0.95
                    st.success("**Výpočet byl úspěšný!**")
                    st.metric("Vzdálenost (km)", f"{res:.3f} km")
                    zapis = f"km = {dil_m:g} m / {dil_dc:g} dc (-5%) = {res:.3f} km"
            elif dil_dc is None:
                if dil_km == 0:
                    st.error("Vzdálenost nesmí být nulová!")
                else:
                    res = (dil_m / dil_km) * 0.95
                    st.success("**Výpočet byl úspěšný!**")
                    st.metric("Úhel (dc)", f"{res:.3f} dc")
                    zapis = f"dc = {dil_m:g} m / {dil_km:g} km (-5%) = {res:.3f} dc"
            if zapis:
                st.session_state.history.append({"Úloha": "Dílcové pravidlo", "Zápis": zapis})
                save_history(st.session_state.history)

# ============================================================
# STRÁNKA: HGÚ 1
# ============================================================
elif st.session_state.page == 'hgu1':
    st.title("I. Hlavní geodetická úloha")
    st.button("Zpět na hlavní menu",     on_click=go_to_home,   use_container_width=True)
    st.button("Vymazat všechny hodnoty", on_click=clear_inputs, use_container_width=True)
    st.markdown("---")

    zobrazit_mapu = st.checkbox(
        "🗺️ Zobrazit geografickou mapu (vyžaduje zadání MGRS oblasti)",
        key="map_hgu1"
    )

    zone_square_hgu1 = None
    if zobrazit_mapu:
        zone_square_hgu1, zs_err = mgrs_zone_input("hgu1")
        if zs_err:
            st.warning(f"⚠️ {zs_err}")
            zone_square_hgu1 = None

    st.markdown("---")
    st.subheader("Zadání hodnot")

    col1, col2 = st.columns(2)
    with col1:
        ea   = st.number_input("E bodu A:",             step=1,              key='ea1')
        na   = st.number_input("N bodu A:",             step=1,              key='na1')
        alta = st.number_input("Alt bodu A:",           step=1,              key='alta1')
    with col2:
        s     = st.number_input("Vzdálenost (m):",      step=1, min_value=0, key='s1')
        angle = st.number_input("Směrník (0–5999 dc):", step=1,              key='ang1')
        pol   = st.number_input("Polohový úhel (dc):",  step=1,              key='pol1')

    if st.button("Vypočítat HGÚ 1", type="primary", use_container_width=True):
        validate_smernik(angle, "Směrník")

        angle_rad = angle * math.pi / 3000.0
        eb   = ea + s * math.sin(angle_rad)
        nb   = na + s * math.cos(angle_rad)
        km   = s / 1000.0
        altb = alta + (pol * km * 1.05)

        st.success("**Výpočet byl úspěšný!**")
        c1, c2, c3 = st.columns(3)
        c1.metric("E bodu B",   f"{eb:.0f}")
        c2.metric("N bodu B",   f"{nb:.0f}")
        c3.metric("Alt bodu B", f"{altb:.0f}")

        zapis = (f"A({ea:.0f}, {na:.0f}, h:{alta:.0f}) ➔ "
                 f"B({eb:.0f}, {nb:.0f}, h:{altb:.0f}) | "
                 f"s={s}, sm={angle}, pol={pol}")
        st.session_state.history.append({"Úloha": "HGÚ 1", "Zápis": zapis})
        save_history(st.session_state.history)

        st.markdown("---")
        st.subheader("Náčrt situace")
        fig = draw_plot(ea, na, eb, nb, angle, s)
        st.pyplot(fig)

        if zobrazit_mapu:
            st.markdown("---")
            st.subheader("Geografická poloha bodů")
            if zone_square_hgu1 is None:
                st.warning("Zadejte platnou MGRS zónu a 100km čtverec.")
            else:
                if not (0 <= eb < 100000) or not (0 <= nb < 100000):
                    st.warning("⚠️ Bod B překračuje hranici 100km čtverce.")
                lat_a, lon_a = mgrs_en_to_wgs84(ea, na, zone_square_hgu1)
                lat_b, lon_b = mgrs_en_to_wgs84(eb, nb, zone_square_hgu1)
                if lat_a is None or lat_b is None:
                    st.error("Nepodařilo se převést souřadnice.")
                else:
                    label_a = (f"Stanovisko | "
                               f"MGRS: {zone_square_hgu1} {int(ea):05d} {int(na):05d}")
                    label_b = (f"Výsledný bod B | "
                               f"MGRS: {zone_square_hgu1} {int(eb):05d} {int(nb):05d}")
                    show_map(lat_a, lon_a, lat_b, lon_b,
                             label_a, label_b, map_key="hgu1")

# ============================================================
# STRÁNKA: HGÚ 2
# ============================================================
elif st.session_state.page == 'hgu2':
    st.title("II. Hlavní geodetická úloha")
    st.button("Zpět na hlavní menu",     on_click=go_to_home,   use_container_width=True)
    st.button("Vymazat všechny hodnoty", on_click=clear_inputs, use_container_width=True)
    st.markdown("---")

    zobrazit_mapu = st.checkbox(
        "🗺️ Zobrazit geografickou mapu (vyžaduje zadání MGRS oblasti)",
        key="map_hgu2"
    )

    zone_square_hgu2 = None
    if zobrazit_mapu:
        zone_square_hgu2, zs_err = mgrs_zone_input("hgu2")
        if zs_err:
            st.warning(f"⚠️ {zs_err}")
            zone_square_hgu2 = None

    st.markdown("---")
    st.subheader("Zadání hodnot")

    col1, col2 = st.columns(2)
    with col1:
        ea   = st.number_input("E bodu A (Stanovisko):",  step=1, key='ea2')
        na   = st.number_input("N bodu A (Stanovisko):",  step=1, key='na2')
        alta = st.number_input("Alt bodu A (Stanovisko):", step=1, key='alta2')
    with col2:
        eb   = st.number_input("E bodu B (Cíl):",  step=1, key='eb2')
        nb   = st.number_input("N bodu B (Cíl):",  step=1, key='nb2')
        altb = st.number_input("Alt bodu B (Cíl):", step=1, key='altb2')

    if st.button("Vypočítat HGÚ 2", type="primary", use_container_width=True):
        de = eb - ea
        dn = nb - na
        s  = math.sqrt(de**2 + dn**2)

        if s == 0:
            st.error("⚠️ Body A a B mají stejné souřadnice!")
            st.stop()

        angle_rad      = math.atan2(de, dn)
        angle_dilce    = (angle_rad * 3000.0 / math.pi) % 6000
        zpetny_smernik = (angle_dilce + 3000) % 6000
        km             = s / 1000.0
        dh             = altb - alta
        polohovy_uhel  = (dh / km) * 0.95

        st.success("**Výpočet byl úspěšný!**")
        c1, c2 = st.columns(2)
        c1.metric("Vzdálenost (m)", f"{s:.0f}")
        c2.metric("Směrník (dc)",   f"{angle_dilce:.0f}")
        c3, c4 = st.columns(2)
        c3.metric("Zpětný směrník",     f"{zpetny_smernik:.0f}")
        c4.metric("Polohový úhel (dc)", f"{polohovy_uhel:.0f}")

        zapis = (f"A({ea:.0f}, {na:.0f}, h:{alta:.0f}) ➔ "
                 f"B({eb:.0f}, {nb:.0f}, h:{altb:.0f}) | "
                 f"s={s:.0f}, sm={angle_dilce:.0f}, pol={polohovy_uhel:.0f}")
        st.session_state.history.append({"Úloha": "HGÚ 2", "Zápis": zapis})
        save_history(st.session_state.history)

        st.markdown("---")
        st.subheader("Náčrt situace")
        fig = draw_plot(ea, na, eb, nb, angle_dilce, s)
        st.pyplot(fig)

        if zobrazit_mapu:
            st.markdown("---")
            st.subheader("Geografická poloha bodů")
            if zone_square_hgu2 is None:
                st.warning("Zadejte platnou MGRS zónu a 100km čtverec.")
            else:
                lat_a, lon_a = mgrs_en_to_wgs84(ea, na, zone_square_hgu2)
                lat_b, lon_b = mgrs_en_to_wgs84(eb, nb, zone_square_hgu2)
                if lat_a is None or lat_b is None:
                    st.error("Nepodařilo se převést souřadnice.")
                else:
                    label_a = (f"Stanovisko | "
                               f"MGRS: {zone_square_hgu2} {int(ea):05d} {int(na):05d}")
                    label_b = (f"Cíl | "
                               f"MGRS: {zone_square_hgu2} {int(eb):05d} {int(nb):05d}")
                    show_map(lat_a, lon_a, lat_b, lon_b,
                             label_a, label_b, map_key="hgu2")
