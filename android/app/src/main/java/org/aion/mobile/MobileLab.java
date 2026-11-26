package org.aion.mobile;

import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Compact Android-only image classification lab. Labels 0..2 are abstract, never actions. */
public final class MobileLab {
    public static final int DIM=256, LABELS=3, PARAMETERS=771, CAPACITY=192;
    private static final int MAGIC=0x414c4231;
    private final List<Sample> samples=new ArrayList<>();
    private float[] weights=new float[PARAMETERS];
    private static final class Sample {
        final float[] x; final int label;
        Sample(float[] x,int label){this.x=x.clone();this.label=label;}
    }
    public static float[] encode(int[] argb,int width,int height) {
        if(width<1||height<1||width>512||height>512||argb.length!=width*height) throw new IllegalArgumentException("invalid image size");
        float[] v=new float[DIM];
        for(int y=0;y<8;y++) for(int x=0;x<8;x++) {
            int x0=x*width/8,x1=Math.max(x0+1,(x+1)*width/8),y0=y*height/8,y1=Math.max(y0+1,(y+1)*height/8);
            int i=y*8+x,n=0; float r=0,g=0,b=0;
            for(int yy=y0;yy<y1;yy++) for(int xx=x0;xx<x1;xx++) {int p=argb[yy*width+xx];r+=(p>>>16)&255;g+=(p>>>8)&255;b+=p&255;n++;}
            v[i*3]=r/(255*n);v[i*3+1]=g/(255*n);v[i*3+2]=b/(255*n);
        }
        for(int i=0;i<64;i++) {int j=i%8==7?i-1:i+1;float a=0,b=0;for(int c=0;c<3;c++){a+=v[3*i+c];b+=v[3*j+c];}v[192+i]=Math.abs(a-b)/3;}
        return v;
    }
    private static void validate(float[] x){if(x.length!=DIM)throw new IllegalArgumentException("256 features required");for(float f:x)if(!Float.isFinite(f)||f<0||f>1)throw new IllegalArgumentException("invalid mobile features");}
    private static boolean held(float[] x){return Math.floorMod(Arrays.hashCode(x),5)==0;}
    public boolean add(float[] x,int label){
        validate(x);if(label<0||label>=LABELS)throw new IllegalArgumentException("invalid label");
        for(Sample s:samples)if(Arrays.equals(s.x,x)){if(s.label!=label)throw new IllegalArgumentException("contradictory label");return false;}
        if(samples.size()==CAPACITY)samples.remove(0);samples.add(new Sample(x,label));return true;
    }
    public int size(){return samples.size();}
    private static float[] predict(float[] w,float[] x){
        float[] p=new float[LABELS];float max=-Float.MAX_VALUE;
        for(int a=0;a<LABELS;a++){float s=w[768+a];for(int j=0;j<DIM;j++)s+=w[a*DIM+j]*x[j];p[a]=s;max=Math.max(max,s);}
        float sum=0;for(int a=0;a<LABELS;a++){p[a]=(float)Math.exp(p[a]-max);sum+=p[a];}for(int a=0;a<LABELS;a++)p[a]/=sum;return p;
    }
    public float[] probabilities(float[] x){validate(x);return predict(weights,x);}
    private static double loss(float[] w,List<Sample> set){double sum=0;for(Sample s:set)sum-=Math.log(Math.max(1e-9,predict(w,s.x)[s.label]));return sum/set.size();}
    private static int correct(float[] w,List<Sample> set){int n=0;for(Sample s:set){float[] p=predict(w,s.x);int best=0;for(int a=1;a<LABELS;a++)if(p[a]>p[best])best=a;if(best==s.label)n++;}return n;}
    public String train(){
        List<Sample> train=new ArrayList<>(),test=new ArrayList<>();int[] a=new int[3],b=new int[3];
        for(Sample s:samples)if(held(s.x)){test.add(s);b[s.label]++;}else{train.add(s);a[s.label]++;}
        for(int i=0;i<3;i++)if(a[i]<4||b[i]<2)return "Datos insuficientes: cada etiqueta requiere 4 ejemplos de entrenamiento y 2 de validacion. Total: "+samples.size();
        float[] candidate=weights.clone();double before=loss(weights,test);int oldCorrect=correct(weights,test);
        for(int epoch=0;epoch<24;epoch++)for(Sample s:train){float[] p=predict(candidate,s.x);for(int k=0;k<3;k++){float d=p[k]-(k==s.label?1:0);for(int j=0;j<DIM;j++)candidate[k*DIM+j]-=.04f*d*s.x[j];candidate[768+k]-=.04f*d;}}
        double after=loss(candidate,test);int newCorrect=correct(candidate,test);boolean accept=Double.isFinite(after)&&after<before-1e-4&&newCorrect>=oldCorrect;
        if(accept)weights=candidate;
        return String.format(Locale.ROOT,"%s | entrenamiento %d, validacion %d | perdida %.5f -> %.5f | aciertos %d -> %d de %d",accept?"Aceptado":"Rechazado",train.size(),test.size(),before,after,oldCorrect,newCorrect,test.size());
    }
    public float[] recall(float[] x){validate(x);Sample best=null;double distance=Double.POSITIVE_INFINITY;for(Sample s:samples){double d=0;for(int i=0;i<DIM;i++){double diff=x[i]-s.x[i];d+=diff*diff;}if(d<distance){distance=d;best=s;}}return best==null?new float[]{-1,-1}:new float[]{best.label,(float)Math.sqrt(distance)};}
    private static byte[] hash(byte[] data)throws IOException{try{return MessageDigest.getInstance("SHA-256").digest(data);}catch(Exception e){throw new IOException(e);}}
    public void save(Path path)throws IOException{
        ByteArrayOutputStream bytes=new ByteArrayOutputStream();try(DataOutputStream out=new DataOutputStream(bytes)){out.writeInt(MAGIC);out.writeInt(samples.size());for(float w:weights)out.writeFloat(w);for(Sample s:samples){out.writeByte(s.label);for(float v:s.x)out.writeFloat(v);}}
        byte[] body=bytes.toByteArray();Path tmp=Files.createTempFile(path.toAbsolutePath().getParent(),"lab-",".tmp");
        try{try(FileOutputStream stream=new FileOutputStream(tmp.toFile())){stream.write(body);stream.write(hash(body));stream.getFD().sync();}Files.move(tmp,path,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);}finally{Files.deleteIfExists(tmp);}
    }
    public static MobileLab load(Path path)throws IOException{
        MobileLab lab=new MobileLab();if(!Files.exists(path))return lab;
        long size=Files.size(path);if(size<8+4*PARAMETERS+32||size>8+4*PARAMETERS+CAPACITY*1025+32)throw new IOException("invalid lab size");
        byte[] all=Files.readAllBytes(path),body=Arrays.copyOf(all,all.length-32);if(!MessageDigest.isEqual(hash(body),Arrays.copyOfRange(all,all.length-32,all.length)))throw new IOException("lab checksum mismatch");
        try(DataInputStream in=new DataInputStream(new ByteArrayInputStream(body))){if(in.readInt()!=MAGIC)throw new IOException("invalid lab version");int n=in.readInt();if(n<0||n>CAPACITY||body.length!=8+PARAMETERS*4+n*1025)throw new IOException("invalid lab count");for(int i=0;i<PARAMETERS;i++){lab.weights[i]=in.readFloat();if(!Float.isFinite(lab.weights[i])||Math.abs(lab.weights[i])>100)throw new IOException("invalid lab weight");}for(int i=0;i<n;i++){int label=in.readUnsignedByte();float[] x=new float[DIM];for(int j=0;j<DIM;j++)x[j]=in.readFloat();if(!lab.add(x,label))throw new IOException("duplicate record");}}
        catch(IllegalArgumentException e){throw new IOException("invalid lab record",e);}return lab;
    }
}
