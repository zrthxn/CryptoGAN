echo "This script will send the desired file to AWS."

read -p "AWS EC2 IP/Host: " ec2ip
read -p "AWS EC2 User: " ec2usr
read -p "File Name (src/): " file

# pemfile = '/mnt/c/Users/User/Desktop/zxaws.pem'

sudo scp -i /mnt/c/Users/User/Desktop/zxaws.pem src/$file $ec2usr@$ec2ip:~/src/$file

# Server Area
sudo ssh $ec2usr@$ec2ip -i /mnt/c/Users/User/Desktop/zxaws.pem

#read -p "Start training? (Y/N) " start

#if [$start == 'Y']
#then
#  python3 $file
#fi

#echo "Quitting"
#logout
# End Server Area

#echo "End"
