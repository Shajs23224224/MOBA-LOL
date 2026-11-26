import org.aion.mobile.MobileLab;
import java.nio.file.*;
import java.util.*;
public class MobileLabProbe {
    public static void main(String[] args)throws Exception{
        MobileLab lab=new MobileLab();Random r=new Random(42);
        for(int n=0;n<150;n++){int label=n%3;int[] pixels=new int[64];for(int i=0;i<64;i++){int p=0xff000000|((180+r.nextInt(76))<<(16-8*label));pixels[i]=p|r.nextInt(16);}lab.add(MobileLab.encode(pixels,8,8),label);}
        float[] x=MobileLab.encode(new int[]{0xffff0000},1,1);float[] old=lab.probabilities(x);
        String report=lab.train();if(!report.startsWith("Aceptado")||Arrays.equals(old,lab.probabilities(x)))throw new AssertionError(report);
        if(lab.recall(x)[0]!=0)throw new AssertionError("recall");
        Path path=Path.of(args[0]);lab.save(path);MobileLab restored=MobileLab.load(path);
        if(restored.size()!=150||!Arrays.equals(lab.probabilities(x),restored.probabilities(x)))throw new AssertionError("restore");
        byte[] bytes=Files.readAllBytes(path);bytes[30]^=1;Files.write(path,bytes);
        try{MobileLab.load(path);throw new AssertionError("corrupt accepted");}catch(java.io.IOException expected){}
        System.out.println(report);System.out.println("replay=150; restore=exact; corruption=rejected; parameters="+MobileLab.PARAMETERS);
    }
}
