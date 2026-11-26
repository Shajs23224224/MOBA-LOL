package org.aion.mobile;
import java.util.*;

/** Role-scoped evidence review, not learned expertise or action execution. */
public final class MobileCouncil {
    public static final int ROLES=160, DEPARTMENTS=16;
    public static final class Evidence {
        public final int role;public final String source;public final double value,confidence,risk;
        public Evidence(int role,String source,double value,double confidence,double risk){if(role<1||role>160||source.isEmpty()||source.length()>128||!Double.isFinite(value)||value< -1||value>1||!(confidence>0&&confidence<=1)||!(risk>=0&&risk<=1))throw new IllegalArgumentException("invalid evidence");this.role=role;this.source=source;this.value=value;this.confidence=confidence;this.risk=risk;}
    }
    public static final class Result {
        public final int evaluated=ROLES,proposals,abstentions,sources;public final boolean contradiction,review;public final double utility,risk;
        Result(int p,int n,boolean c,double u,double r){proposals=p;abstentions=ROLES-p;sources=n;contradiction=c;utility=u;risk=r;review=true;}
    }
    public static Result deliberate(List<Evidence> evidence){if(evidence.size()>4096)throw new IllegalArgumentException("evidence budget");Map<String,double[]> groups=new TreeMap<>();int proposals=0;boolean pos=false,neg=false;double risk=0;for(int role=1;role<=ROLES;role++){boolean active=false;for(Evidence e:evidence)if(e.role==role){active=true;double[] g=groups.computeIfAbsent(e.source,k->new double[3]);g[0]+=e.value*e.confidence;g[1]+=e.confidence;g[2]=Math.max(g[2],e.confidence);risk=Math.max(risk,e.risk);pos|=e.value>=.4&&e.confidence>=.7;neg|=e.value<=-.4&&e.confidence>=.7;}if(active)proposals++;}double total=0,weight=0;for(double[] g:groups.values()){total+=g[0]/g[1]*g[2];weight+=g[2];}return new Result(proposals,groups.size(),pos&&neg,weight==0?0:total/weight,risk);}
    public static Set<String> closure(Set<String> facts,Map<String,Set<String>> rules){if(facts.size()>1024||rules.size()>128)throw new IllegalArgumentException("rule budget");Set<String> known=new HashSet<>(facts);boolean change;do{change=false;for(Map.Entry<String,Set<String>> rule:rules.entrySet())if(known.containsAll(rule.getValue()))change|=known.add(rule.getKey());}while(change);return known;}
    public static double belief(double prior,boolean present,double reliability){if(!(prior>0&&prior<1)||!(reliability>.5&&reliability<1))throw new IllegalArgumentException("invalid probability");double likelihood=present?reliability:1-reliability;return prior*likelihood/(prior*likelihood+(1-prior)*(1-likelihood));}
}
