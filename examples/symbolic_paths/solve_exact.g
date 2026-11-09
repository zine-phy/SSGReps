# Run in a case directory containing group_data.g:
# gap -q -A -b --quitonbreak ../solve_exact.g
# Exact +/-1 projective coreps: Wigner +1 and 0 are needed by these ten cases.
Read("group_data.g");;
SetAssertionLevel(1);;
started:=Runtime();;
Conj:=m -> List(m,row -> List(row,ComplexConjugate));;
Adj:=m -> TransposedMat(Conj(m));;
Dense:=m -> List([1..Length(m)],i -> List([1..Length(m[1])],j -> m[i][j]));;
SqrtQ:=x -> ER(NumeratorRat(x))/ER(DenominatorRat(x));;
perms:=List([1..h],g -> PermList(Concatenation(
  List([1..h],x -> mt[x][g]+h*ce[x][g]),
  List([1..h],x -> mt[x][g]+h*((1+ce[x][g]) mod 2)))));;
z:=PermList(Concatenation([h+1..2*h],[1..h]));;
ext:=Group(z);; gens:=[];;
for i in [1..h] do
 if not perms[i] in ext then Add(gens,i); ext:=Group(Concatenation([z],perms{gens})); fi;
od;
Assert(1,Size(ext)=2*h);
classes:=ConjugacyClasses(ext);;
classOf:=List(perms,p -> PositionProperty(classes,c -> p in c));;
zc:=PositionProperty(classes,c -> z in c);;
chars:=Filtered(Irr(ext),c -> c[zc]=-c[1]);;
Wigner:=c -> Sum([h+1..n],x -> (-1)^ce[x][x]*c[classOf[mt[x][x]]])/h;;
Print("order=",n," H=",h," irreps(dim,W)=",List(chars,c -> [c[1],Wigner(c)]),"\n");
Assert(1,Sum(chars,c -> c[1]^2)=h);

BuildD:=function(chi)
 local B,C,t,tblB,irrB,res,theta,trial,rhoB,ind,ambient,P,rows,classesB,
       weight,Q,v,b,norm,col,basis,D,i,weights;
 Reset(GlobalMersenneTwister,2218363+chi[1]);
 theta:=fail;
 for trial in [1..100] do
  B:=Group(z); C:=ext;
  while Size(C)>Size(B) do
   repeat t:=Random(C); until not t in B;
   B:=Group(Concatenation(GeneratorsOfGroup(B),[t])); C:=Centralizer(ext,B);
  od;
  tblB:=CharacterTable(B); irrB:=Irr(tblB);
  FusionConjugacyClasses(tblB,CharacterTable(ext)); res:=Restricted(chi,tblB);
  theta:=First(irrB,x -> x[1]=1 and ScalarProduct(res,x)=1);
  if theta<>fail then break; fi;
 od;
 Assert(1,theta<>fail);
 rhoB:=IrreducibleRepresentationsDixon(B,theta);
 ind:=InducedRepresentation(rhoB,ext); ambient:=Index(ext,B);
 P:=chi[1]/Size(ext)*Sum([1..h],i -> 2*ComplexConjugate(chi[classOf[i]])*Dense(Image(ind,perms[i])));
 Assert(1,P*P=P and Adj(P)=P);
 rows:=[];classesB:=ConjugacyClasses(B);
 for weight in irrB do
  if ScalarProduct(res,weight)>0 then
   Q:=P*Sum([1..Length(classesB)],i -> ComplexConjugate(weight[i])*Dense(Image(ind,Representative(classesB[i]))))/Size(B);
   for col in TransposedMat(Q) do
    v:=ShallowCopy(col);
    for b in rows do v:=v-(List(b,ComplexConjugate)*v)*b; od;
    norm:=List(v,ComplexConjugate)*v;
    if norm<>0 then
     Assert(1,IsRat(norm) and norm>0); Add(rows,v/SqrtQ(norm));
    fi;
   od;
  fi;
 od;
 basis:=TransposedMat(rows);
 Assert(1,Length(rows)=chi[1] and Adj(basis)*basis=IdentityMat(chi[1]));
 D:=List(perms,p -> Adj(basis)*Dense(Image(ind,p))*basis);
 Assert(1,List(D,TraceMat)=List(classOf,i -> chi[i]));
 return D;
end;;

CorepBlocks:=function(X,Y)
 local d,M,i,j;
 d:=Length(X); M:=NullMat(2*d,2*d);
 for i in [1..d] do for j in [1..d] do M[i][j]:=X[i][j]; M[i+d][j+d]:=Y[i][j]; od; od;
 return M;
end;;

a:=h+1;; ainv:=Position(mt[a],1);; a2:=mt[a][a];;
used:=[];; results:=[];; D:=[];; Bmats:=[];; ProbeMat:=[];; partner:=[];; character:=[];; corepDimension:=0;;
for chi in chars do
 charH:=List(classOf,i -> chi[i]);
 if charH in used then continue; fi;
 D:=BuildD(chi); d:=chi[1]; w:=Wigner(chi);
 Assert(1,w=1 or w=0); # Unsupported cases fail instead of silently doubling.
 if w=1 then
  # Orthogonal projection onto Hom(conj(D), c_h D(a h a^-1)).
  Bmats:=List([1..h],i -> (-1)^(ce[a][i]-ce[mt[mt[a][i]][ainv]][a])*D[mt[mt[a][i]][ainv]]);
  U:=NullMat(d,d);
  for row in [1..d] do
   for col in [1..d] do
    ProbeMat:=NullMat(d,d); ProbeMat[row][col]:=1;
    U:=Sum([1..h],i -> Bmats[i]*ProbeMat*TransposedMat(D[i]));
    if U<>NullMat(d,d) then break; fi;
   od;
   if U<>NullMat(d,d) then break; fi;
  od;
  Assert(1,U<>NullMat(d,d));
  metric:=Adj(U)*U; q:=metric[1][1];
  Assert(1,metric=q*IdentityMat(d) and IsRat(q) and q>0);
  MA:=U/SqrtQ(q);
  Assert(1,MA*Conj(MA)=(-1)^ce[a][a]*D[a2]);
  M:=ShallowCopy(D); corepDimension:=d; Add(used,charH);
 else
  partner:=List([1..h],i -> (-1)^(ce[i][a]-ce[a][mt[mt[ainv][i]][a]])*Conj(D[mt[mt[ainv][i]][a]]));
  M:=List([1..h],i -> CorepBlocks(D[i],partner[i]));
  MA:=NullMat(2*d,2*d);
  for i in [1..d] do for j in [1..d] do MA[i][j+d]:=(-1)^ce[a][a]*D[a2][i][j]; od; MA[i+d][i]:=1; od;
  corepDimension:=2*d; Add(used,charH); Add(used,List(partner,TraceMat));
 fi;
 for i in [1..h] do M[mt[a][i]]:=(-1)^ce[a][i]*MA*Conj(M[i]); od;
 character:=List(M{[1..h]},TraceMat);
 norm:=Sum(character,x -> x*ComplexConjugate(x))/h;
 antiSum:=Sum([h+1..n],x -> (-1)^ce[x][x]*character[mt[x][x]])/h;
 Assert(1,(norm+antiSum)/2=1);
 Assert(1,ForAll(M,x -> x*Adj(x)=IdentityMat(corepDimension)));
 for i in [1..n] do
  for j in [1..n] do
   if i<=h then rhs:=M[j]; else rhs:=Conj(M[j]); fi;
   Assert(1,M[i]*rhs=(-1)^ce[i][j]*M[mt[i][j]]);
  od;
 od;
 Add(results,rec(dimension:=corepDimension,wigner:=w,character:=character,
                 restrictionNorm:=norm,matrices:=M,entries:=Set(Flat(M))));
 Print("PASS dimension=",corepDimension," W=",w," conductor=",Conductor(Set(Flat(M)))," products=",n*n,"\n");
od;
Assert(1,Length(used)=Length(chars));
out:=OutputTextFile("representations.json",false);; SetPrintFormattingStatus(out,false);
PrintTo(out,"{\"order\":",n,",\"unitary_order\":",h,",\"generator_indices_0based\":",List(gens,i -> i-1),
 ",\"anti_generator_index_0based\":",a-1,",\"representations\":[");
for i in [1..Length(results)] do
 r:=results[i]; if i>1 then PrintTo(out,","); fi;
 PrintTo(out,"{\"label\":\"r",i,"\",\"dimension\":",r.dimension,",\"wigner\":",r.wigner,
  ",\"restriction_norm\":",r.restrictionNorm,",\"irreducibility_indicator\":1,\"exact_products_checked\":",n*n,
  ",\"conductor\":",Conductor(r.entries),",\"entries\":",List(r.entries,String),
  ",\"character\":",List(r.character,String),",\"matrices\":",List(r.matrices,m -> List(m,row -> List(row,String))),"}");
od;
PrintTo(out,"]}\n"); CloseStream(out);
# A compact GAP file allows exact compatibility relations between endpoints
# and lines, without reparsing any floating point data.
PrintTo("characters.g","characters:=",List(results,r -> r.character),";;\ndimensions:=",List(results,r -> r.dimension),";;\n");
Print("Finished ",Length(results)," irreducible coreps; seconds=",(Runtime()-started)/1000.,"\n");
QUIT;
