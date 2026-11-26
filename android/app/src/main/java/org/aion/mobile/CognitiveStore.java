package org.aion.mobile;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Eight bounded categories per private session file; observations are not inferred facts. */
public final class CognitiveStore {
    public static final String[] KINDS={"sensory","working","episodic","semantic","procedural","strategic","predictive","metacognitive"};
    private final List<List<Record>> records=new ArrayList<>();
    public static final int CAP=32;
    public static final class Record {
        public final long tick;public final String text;private final float[] vector;
        Record(long t,String s,float[] v){tick=t;text=s;vector=v.clone();}
    }
    public CognitiveStore(){for(int i=0;i<8;i++)records.add(new ArrayList<>());}
    private static void check(float[] v){if(v.length!=256)throw new IllegalArgumentException("256 features required");for(float x:v)if(!Float.isFinite(x)||Math.abs(x)>10000)throw new IllegalArgumentException("invalid features");}
    public void put(int kind,long tick,String text,float[] v){check(v);if(kind<0||kind>=8||tick<0||text.length()>1024)throw new IllegalArgumentException("invalid record");List<Record> list=records.get(kind);if(!list.isEmpty()&&tick<list.get(list.size()-1).tick)throw new IllegalArgumentException("stale tick");if(list.size()==CAP)list.remove(0);list.add(new Record(tick,text,v));}
    public int count(int kind){return records.get(kind).size();}
    public Record recall(int kind,float[] v){check(v);double best=Double.POSITIVE_INFINITY;Record result=null;for(Record r:records.get(kind)){double d=0;for(int i=0;i<256;i++){double x=r.vector[i]-v[i];d+=x*x;}if(d<=best){best=d;result=r;}}return result;}
    public void save(Path path)throws IOException{Path tmp=Files.createTempFile(path.toAbsolutePath().getParent(),"cognition-",".tmp");try{try(FileOutputStream stream=new FileOutputStream(tmp.toFile());DataOutputStream out=new DataOutputStream(stream)){out.writeInt(0x41434d31);for(List<Record> list:records){out.writeInt(list.size());for(Record r:list){out.writeLong(r.tick);out.writeUTF(r.text);for(float x:r.vector)out.writeFloat(x);}}out.flush();stream.getFD().sync();}Files.move(tmp,path,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);}finally{Files.deleteIfExists(tmp);}}
    public static CognitiveStore load(Path path)throws IOException{CognitiveStore s=new CognitiveStore();if(!Files.exists(path))return s;if(Files.size(path)>1200000)throw new IOException("store too large");try(DataInputStream in=new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))){if(in.readInt()!=0x41434d31)throw new IOException("invalid store");for(int k=0;k<8;k++){int n=in.readInt();if(n<0||n>CAP)throw new IOException("invalid count");for(int i=0;i<n;i++){long t=in.readLong();String text=in.readUTF();float[] v=new float[256];for(int j=0;j<256;j++)v[j]=in.readFloat();s.put(k,t,text,v);}}if(in.read()!=-1)throw new IOException("trailing store data");}catch(IllegalArgumentException e){throw new IOException("invalid record",e);}return s;}
}
