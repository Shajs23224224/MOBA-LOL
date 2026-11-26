package org.aion.mobile;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.*;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/** Explicit diagnostic imports, no background capture or game controls. */
public final class CognitiveActivity extends Activity {
    private static final ExecutorService WORK=Executors.newSingleThreadExecutor();
    private TextView status; private EditText id,note; private LinearLayout layout;
    private float[] vector,actor; private NeuralPort fusion; private boolean busy; private Path store;
    private final Map<Integer,float[]> experts=new HashMap<>();
    private int requestedId;
    @Override public void onCreate(Bundle state){super.onCreate(state);store=getFilesDir().toPath().resolve("cognition-v1.bin");
        vector=getIntent().getFloatArrayExtra("features");actor=getIntent().getFloatArrayExtra("actor");if(vector==null||vector.length!=256)vector=new float[256];if(actor==null||actor.length!=4)actor=new float[4];
        ScrollView scroll=new ScrollView(this);layout=new LinearLayout(this);layout.setOrientation(LinearLayout.VERTICAL);layout.setPadding(24,40,24,24);scroll.addView(layout);setContentView(scroll);
        status=new TextView(this);status.setText("Integracion diagnostica de 34 redes. Vector de la pantalla anterior. Sin control del juego ni confianza calibrada.");layout.addView(status);
        button("Importar fusion AFN1",()->pick(1));id=new EditText(this);id.setInputType(2);id.setHint("Experto N001–N824 (numero)");id.setText("1");layout.addView(id);
        button("Importar y evaluar experto AEN1",()->{try{requestedId=Integer.parseInt(id.getText().toString());if(requestedId<1||requestedId>824)throw new IllegalArgumentException();pick(2);}catch(Exception e){status.setText("ID debe estar entre 1 y 824");}});
        button("Ejecutar diagnostico cognitivo",()->runFusion());
        note=new EditText(this);note.setHint("Nota de investigacion (max. 1024 caracteres)");layout.addView(note);
        button("Guardar conocimiento semantico",()->saveNote(3));button("Guardar procedimiento",()->saveNote(4));
    }
    private void button(String label,Runnable run){Button b=new Button(this);b.setText(label);b.setOnClickListener(v->{if(!busy)run.run();});layout.addView(b);}
    private void pick(int request){Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("*/*");startActivityForResult(i,request);}
    private void finish(String text){runOnUiThread(()->{if(!isDestroyed()){busy=false;status.setText(text);}});}
    private void task(Callable<String> f){busy=true;status.setText("Procesando localmente...");WORK.execute(()->{try{finish(f.call());}catch(Exception e){finish("Error: "+e.getMessage());}});}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(result!=RESULT_OK||data==null||data.getData()==null||busy)return;Uri uri=data.getData();int selected=requestedId;
        task(()->{try(InputStream stream=getContentResolver().openInputStream(uri)){if(stream==null)throw new IllegalArgumentException("archivo no disponible");if(request==1){fusion=NeuralPort.load(stream);return "34 redes de fusion importadas; hash verificado.";}if(request==2){float[] output=NeuralPort.expert(stream,selected,vector);int group=selected<=416?(selected-1)/52:8+(selected-417)/51;if(experts.size()>=2&&!experts.containsKey(group))experts.clear();experts.put(group,output);return "Experto N"+selected+" evaluado; subsistemas disponibles: "+experts.size();}return "Solicitud desconocida";}});
    }
    private void saveNote(int kind){String value=note.getText().toString().trim();if(value.isEmpty()||value.length()>1024){status.setText("Nota invalida");return;}task(()->{CognitiveStore s=CognitiveStore.load(store);s.put(kind,System.currentTimeMillis(),value,vector);s.save(store);return "Nota guardada; categoria: "+CognitiveStore.KINDS[kind];});}
    private void runFusion(){task(()->{if(fusion==null)return "Importa primero las redes de fusion.";CognitiveStore s=CognitiveStore.load(store);float[][] summaries=new float[16][32];for(int i=0;i<16;i++)for(int k=0;k<8;k++)summaries[i][2*k]=s.recall(k,vector)!=null?1:0;
        MobileCouncil.Result r=MobileCouncil.deliberate(Collections.emptyList());float[] y=fusion.fuse(vector,summaries,experts,actor);long tick=System.currentTimeMillis();for(int kind:new int[]{0,1,2,5,6,7})s.put(kind,tick,"Diagnostico no calibrado; sin evidencia observada",vector);s.save(store);
        return "34 redes evaluadas. Salidas: "+y.length+"\nRoles consultados: "+r.evaluated+"; abstenciones: "+r.abstentions+"\nF02: "+Arrays.toString(Arrays.copyOfRange(y,1104,1107))+"\nMemoria: "+Arrays.toString(new int[]{s.count(0),s.count(1),s.count(2),s.count(3),s.count(4),s.count(5),s.count(6),s.count(7)})+"\nRevision obligatoria; sin acciones.";});}
}
