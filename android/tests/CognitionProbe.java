import org.aion.mobile.*;
import java.util.*;
import java.nio.file.*;
public class CognitionProbe {
    public static void main(String[] args)throws Exception{
        Path path=Path.of(args[0]);CognitiveStore s=new CognitiveStore();float[] x=new float[256];x[0]=1;
        for(int k=0;k<8;k++)for(int i=0;i<40;i++)s.put(k,i,"record-"+i,x);
        s.save(path);s=CognitiveStore.load(path);for(int k=0;k<8;k++)if(s.count(k)!=32||!s.recall(k,x).text.equals("record-39"))throw new AssertionError("memory");
        List<MobileCouncil.Evidence> e=new ArrayList<>();for(int i=1;i<=160;i++)e.add(new MobileCouncil.Evidence(i,"one",.8,.9,.1));e.add(new MobileCouncil.Evidence(1,"two",-.8,.9,.2));
        MobileCouncil.Result r=MobileCouncil.deliberate(e);if(r.evaluated!=160||r.proposals!=160||r.sources!=2||!r.contradiction||Math.abs(r.utility)>1e-9)throw new AssertionError("council");
        if(MobileCouncil.deliberate(Collections.emptyList()).abstentions!=160)throw new AssertionError("abstention");
        if(!MobileCouncil.closure(Set.of("a"),Map.of("b",Set.of("a"),"c",Set.of("b"))).contains("c"))throw new AssertionError("logic");
        if(Math.abs(MobileCouncil.belief(.5,true,.9)-.9)>1e-9)throw new AssertionError("belief");
        System.out.println("memory=8x32; restore=ok; roles=160; contradiction=blocked for review; logic=ok; belief=ok");
    }
}
