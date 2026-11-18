"""Interactive Folium map for observed GPS and matched road routes."""
from pathlib import Path
import folium
from folium.plugins import MiniMap
import numpy as np
from .data import Trip
from .matcher import CandidateBatch


def create_map(trip: Trip, batch: CandidateBatch, predicted: np.ndarray, out_path: Path, max_gps_points=2500):
    center=[float(np.mean(trip.track[:,1])),float(np.mean(trip.track[:,0]))]
    m=folium.Map(location=center,zoom_start=14,control_scale=True,tiles="OpenStreetMap")
    MiniMap(toggle_display=True).add_to(m)
    gps_layer=folium.FeatureGroup(name="Observed GPS",show=True)
    stride=max(1,len(trip.track)//max_gps_points)
    for i in range(0,len(trip.track),stride):
        lon,lat=trip.track[i,:2]
        folium.CircleMarker([lat,lon],radius=2,color="#1976d2",fill=True,fill_opacity=.65,
                            tooltip=f"GPS observation {i}").add_to(gps_layer)
    gps_layer.add_to(m)
    # Display a compact local road context from arcs that appeared in candidate sets.
    context=folium.FeatureGroup(name="Candidate road segments",show=False)
    relevant=np.unique(batch.ids)
    for aid in relevant:
        if aid<0 or aid>=len(trip.arcs):continue
        u,v=trip.arcs[aid].astype(int)
        if min(u,v)<0 or max(u,v)>=len(trip.nodes):continue
        folium.PolyLine([[trip.nodes[u,1],trip.nodes[u,0]],[trip.nodes[v,1],trip.nodes[v,0]]],
                        color="#9e9e9e",weight=2,opacity=.35).add_to(context)
    context.add_to(m)
    def route_layer(name,ids,color,show):
        layer=folium.FeatureGroup(name=name,show=show)
        points=[]
        for aid in ids:
            if aid<0 or aid>=len(trip.arcs):continue
            u,v=trip.arcs[int(aid)].astype(int)
            if min(u,v)<0 or max(u,v)>=len(trip.nodes):continue
            points.append([float(trip.nodes[u,1]),float(trip.nodes[u,0])])
            points.append([float(trip.nodes[v,1]),float(trip.nodes[v,0])])
        if points:folium.PolyLine(points,color=color,weight=5,opacity=.88,tooltip=name).add_to(layer)
        layer.add_to(m)
    route_layer("Reference route",trip.route,"#2e7d32",False)
    route_layer("Predicted route",predicted,"#e53935",True)
    folium.LayerControl(collapsed=False).add_to(m)
    out_path.parent.mkdir(parents=True,exist_ok=True); m.save(str(out_path)); return out_path
