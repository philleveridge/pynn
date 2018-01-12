#
if [ "$1" == "" ]; 
then echo "Must give commit message"; 
else
#add, commit and push
rm *.pyc
git add *
git commit -m "$1"
git push origin master

fi
