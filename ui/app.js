const el = id => document.getElementById(id);
let result, view = [0,0,900,600], drag;
const updateView = () => el("map").setAttribute("viewBox",view.join(" "));
function zoom(factor){
  if(view[2]*factor<10 || view[2]*factor>7200)return;
  view=[view[0]+view[2]*(1-factor)/2,view[1]+view[3]*(1-factor)/2,view[2]*factor,view[3]*factor];updateView();
}
el("plus").onclick=()=>zoom(.7);el("minus").onclick=()=>zoom(1/.7);
el("fit").onclick=()=>{view=[0,0,900,600];updateView()};
const mapPoint=event=>new DOMPoint(event.clientX,event.clientY).matrixTransform(el("map").getScreenCTM().inverse());
el("map").onpointerdown=event=>{drag=mapPoint(event);el("map").setPointerCapture(event.pointerId)};
el("map").onpointermove=event=>{if(!drag)return;const next=mapPoint(event);view[0]+=drag.x-next.x;view[1]+=drag.y-next.y;updateView()};
el("map").onpointerup=el("map").onpointercancel=()=>drag=null;
function draw(){
  if(!result)return;
  const cosine=Math.cos(result.gps[0][1]*Math.PI/180);
  const points=result.gps.concat(result.sections.flatMap(section=>section.coordinates));
  let west=Infinity,east=-Infinity,north=-Infinity,south=Infinity;
  for(const [lon,lat] of points){west=Math.min(west,lon);east=Math.max(east,lon);south=Math.min(south,lat);north=Math.max(north,lat)}
  const width=(east-west)*cosine,height=north-south;
  const scale=Math.min(840/Math.max(width,1e-8),540/Math.max(height,1e-8));
  const xy=([lon,lat])=>[(lon-west)*cosine*scale+(900-width*scale)/2,(north-lat)*scale+(600-height*scale)/2].map(value=>value.toFixed(3)).join(" ");
  const path=(shape,stroke,weight)=>{const node=document.createElementNS("http://www.w3.org/2000/svg","path");
    for(const [name,value] of Object.entries({d:shape,stroke,"stroke-width":weight,fill:"none","stroke-linecap":"round","vector-effect":"non-scaling-stroke"}))node.setAttribute(name,value);
    el("drawing").append(node)};
  el("drawing").replaceChildren();
  if(el("networkToggle").checked)path(result.roads.map(segment=>"M"+segment.map(xy).join(" L")).join(" "),"#b1c1b5",1);
  if(el("roadToggle").checked)for(const section of result.sections)path("M"+section.coordinates.map(xy).join(" L"),"#087c50",4);
  if(el("gpsToggle").checked)path(result.gps.map(point=>"M"+xy(point)+"h.001").join(" "),"#d47627",2);
}
el("networkToggle").onchange=el("roadToggle").onchange=el("gpsToggle").onchange=draw;
async function request(url,method){const response=await fetch(url,{method});if(!response.ok)throw Error("Server returned "+response.status);return response.json()}
el("run").onclick=async()=>{
  const trip=el("trip").value;el("run").disabled=el("trip").disabled=true;
  el("status").textContent="Running classifier and decoder for "+trip+"…";
  try{result=await request("/api/predict/"+encodeURIComponent(trip),"POST");el("fit").click();draw();el("empty").hidden=true;
    el("accuracy").textContent=result.accuracy===null?"Not scored":(100*result.accuracy).toFixed(2)+"%";
    el("count").textContent=result.gps.length.toLocaleString();el("sections").textContent=result.sections.length;
    el("runtime").textContent=result.inference_seconds.toFixed(2)+" s";el("status").textContent="Showing trip "+result.trip+" · live inference complete";
  }catch(error){el("status").textContent="Request failed: "+error.message;el("drawing").replaceChildren();result=null;el("empty").hidden=false;
    for(const id of ["count","sections","accuracy","runtime"])el(id).textContent="—"}
  finally{el("run").disabled=el("trip").disabled=false}
};
request("/api/trips","GET").then(({trips})=>{for(const id of trips)el("trip").add(new Option(id,id));el("status").textContent=trips.length+" trips available";el("run").disabled=!trips.length})
.catch(error=>el("status").textContent="Cannot load trips. Check the API connection and reload. "+error.message);
