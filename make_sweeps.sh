#!/bin/bash

cd $1
parentname="$(basename "$PWD")"

for FILE in ./*.yml; do
  name=$(basename $FILE | cut -d "." -f 1);
  if [[ $name == _* ]]
  then
    echo "Skipping $FILE";
  else
    export sname=$parentname-$name;
    wandb sweep $FILE --name=$sname &> runinfo.txt ;
    SID=$( cat runinfo.txt | sed '2q;d' | cut -c 31-);
    echo $sname: $SID;
    echo "wandb agent $WANDB_USER/$WANDB_PROJECT/$SID #$sname" >> agents.txt;
    mv $name.yml _$name.yml
  fi
done
echo "" >> agents.txt

rm runinfo.txt
rmdir wandb